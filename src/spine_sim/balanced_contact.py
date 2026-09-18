"""Intermediate fidelity: condensed 3-D tapered beam, moving contact, friction history.

Small-rotation Euler-Bernoulli force/moment compliance at the current exposed
length replaces the nonlinear rod. A smooth sphere envelope replaces individual
triangle supports. Axial spring compression omits the configurational force.
This is a separate approximate model, not the strict IJMS solver.
"""
from functools import lru_cache
import numpy as np
import time
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares,minimize

MODEL_VERSION='ijms-balanced-condensed-contact-6'


class EnvelopeSurface:
    def __init__(self,height,dx,dy,origin):
        self.x=origin[0]+np.arange(height.shape[1])*dx
        self.y=origin[1]+np.arange(height.shape[0])*dy
        self.spline=RectBivariateSpline(self.y,self.x,height)
        self.slope_x=self.spline.partial_derivative(0,1)
        self.slope_y=self.spline.partial_derivative(1,0)

    def query(self,xy):
        x,y=xy[:,0],xy[:,1]
        if x.min()<self.x[0] or x.max()>self.x[-1] or y.min()<self.y[0] or y.max()>self.y[-1]:
            raise ValueError('SURFACE_DOMAIN_LIMIT')
        height=self.spline.ev(y,x)
        normal=np.column_stack((-self.slope_x(y,x,grid=False),-self.slope_y(y,x,grid=False),np.ones(len(x))))
        normal/=np.linalg.norm(normal,axis=1)[:,None]
        return height,normal


@lru_cache(maxsize=128)
def compliance_table(p):
    lengths=np.linspace(.000001,p.free_length_m,401)
    nodes,weights=np.polynomial.legendre.leggauss(32)
    values=np.zeros((len(lengths),3))
    for i,length in enumerate(lengths):
        cuts=sorted(set([0.,min(p.taper_length_m,length),length]))
        for low,high in zip(cuts[:-1],cuts[1:]):
            u=(low+high)/2+(high-low)/2*nodes
            w=(high-low)/2*weights/(p.young_modulus_Pa*p.section_second_moment_m4(u))
            values[i]=values[i]+np.array([np.sum(w*u*u),np.sum(w*u),np.sum(w)])
    return lengths,values


class CondensedArray:
    def __init__(self,rods,surface,mu_static=.5,mu_kinetic=.4,rigid=False,
                 friction_transition_m=1e-6,tolerance=1e-3):
        self.rods=rods;self.surface=surface;self.n=len(rods)
        self.axes=np.array([r.axis for r in rods]);self.radius=np.array([r.parameters.tip_radius_m for r in rods])
        self.length=np.array([r.parameters.free_length_m for r in rods])
        self.guides=np.array([r.guide_position_m for r in rods])
        self.spring=np.array([1/r.parameters.spring_stiffness_N_per_m if r.parameters.mount_type=='spring' else 0 for r in rods])
        # A hard stop exists only before the needle disappears into the guide.
        self.compression_stop=np.array([r.parameters.max_compression_m
            if r.parameters.mount_type=='spring' and r.parameters.max_compression_m<r.parameters.free_length_m
            else np.inf for r in rods])
        self.rigid=rigid
        self.tables=[] if rigid else [compliance_table(r.parameters) for r in rods]
        self.fixed_coeff=None if rigid or np.any(self.spring) else np.array([table[1][-1] for table in self.tables])
        self.mu_static=mu_static;self.mu_kinetic=mu_kinetic
        self.friction_transition_m=friction_transition_m;self.tolerance=tolerance

    def deform(self,force,position,normal):
        axial=np.sum(force*self.axes,axis=1)
        # s=min(-Fa/k,smax) is the compression-stop complementarity law:
        # at smax the stop carries -Fa-k*smax; unloading releases it.
        # Do not clamp the lower end: that would create an unrequested shoulder.
        compression=np.minimum(-axial*self.spring,self.compression_stop)
        exposed=self.length-compression
        if self.rigid:
            center=self.guides+position+exposed[:,None]*self.axes
            return center,np.zeros_like(force),compression
        coeff=self.fixed_coeff
        if coeff is None:
            coeff=np.array([[np.interp(l,table[0],table[1][:,j]) for j in range(3)]
                            for l,table in zip(exposed,self.tables)])
        moment=-self.radius[:,None]*np.cross(normal,force)
        transverse=force-axial[:,None]*self.axes
        moment-=np.sum(moment*self.axes,axis=1)[:,None]*self.axes
        center=(self.guides+position+exposed[:,None]*self.axes
                +coeff[:,0,None]*transverse-coeff[:,1,None]*np.cross(self.axes,moment))
        rotation=coeff[:,1,None]*np.cross(self.axes,force)+coeff[:,2,None]*moment
        return center,rotation,compression

    def solve(self,x,preload,previous,guess=None,tolerance=None,max_nfev=80,recover=False,*,fixed_y=None,fixed_z=None):
        if recover and (fixed_y is not None or fixed_z is not None):
            raise ValueError('Fixed-coordinate recovery is not supported')
        if tolerance is None:tolerance=self.tolerance
        force_scale=preload/self.n;length_scale=1e-5
        if guess is None:
            guess=np.r_[previous['force'].ravel()/force_scale,previous['position'][1:]/length_scale]
            if not np.any(previous['force']):
                from .trend_array import distribute_preload,vertical_stiffness
                height,_=self.surface.query(previous['center'][:,:2])
                z,loads=distribute_preload(height[None,:],vertical_stiffness(self.rods),preload)
                guess[:-2]=0;guess[2:-2:3]=loads[0]/force_scale;guess[-1]=z[0]/length_scale
        def evaluate(v):
            force=v[:-2].reshape(-1,3)*force_scale
            position=np.r_[x,v[-2:]*length_scale]
            center,rotation,compression=self.deform(force,position,previous['normal'])
            height,normal=self.surface.query(center[:,:2])
            # One moment-normal correction; its remaining mismatch is reported.
            if not self.rigid:
                center,rotation,compression=self.deform(force,position,normal)
                height,normal=self.surface.query(center[:,:2])
            gap=(center[:,2]-self.radius-height)*normal[:,2]
            normal_force=np.sum(force*normal,axis=1)
            tangent_force=force-normal_force[:,None]*normal
            movement=center-previous['center']+np.cross(rotation-previous['rotation'],-self.radius[:,None]*normal)
            movement-=np.sum(movement*normal,axis=1)[:,None]*normal
            # Resolve the static-to-kinetic jump over 1 um of incremental slip.
            # At zero motion the static disk remains set-valued; developed
            # sliding uses mu_k. This is an explicit numerical approximation.
            mu=self.mu_kinetic+(self.mu_static-self.mu_kinetic)*np.exp(-np.sum(movement*movement,axis=1)/self.friction_transition_m**2)
            trial=tangent_force-force_scale/length_scale*movement
            trial_norm=np.linalg.norm(trial,axis=1)
            limit=mu*np.maximum(normal_force,0.)
            projection=trial*np.minimum(1.,limit/np.maximum(trial_norm,1e-30))[:,None]
            contact=(normal*np.minimum(normal_force/force_scale,gap/length_scale)[:,None]
                     +(tangent_force-projection)/force_scale)
            ry=force[:,1].sum()/preload if fixed_y is None else (position[1]-fixed_y)/length_scale
            rz=(force[:,2].sum()-preload)/preload if fixed_z is None else (position[2]-fixed_z)/length_scale
            residual=np.r_[contact.ravel(),ry,rz]
            return residual,dict(force=force,position=position,center=center,rotation=rotation,compression=compression,
                                 normal=normal,gap=gap,normal_force=normal_force,
                                 slip=(normal_force>force_scale*tolerance)&(trial_norm>limit+force_scale*tolerance),
                                 movement=movement)
        cached_v=None;cached_result=None
        def full(v):
            nonlocal cached_v,cached_result
            if cached_v is None or not np.array_equal(v,cached_v):
                cached_result=evaluate(v);cached_v=v.copy()
            return cached_result
        def residual(v):
            try:return full(v)[0]
            except ValueError as exc:
                if str(exc)!='SURFACE_DOMAIN_LIMIT':raise
                return np.full(3*self.n+2,1e6)
        def jacobian(v):
            initial=full(v)[0];jac=np.zeros((3*self.n+2,len(v)));h=2e-5
            # Rods are independent given the backplate. Perturb the same force
            # component on every rod at once: five evaluations, not 3N+2.
            for j in range(3):
                pert=v.copy();pert[j:-2:3]+=h
                diff=(residual(pert)[:-2]-initial[:-2]).reshape(-1,3)/h
                for i in range(self.n):jac[3*i:3*i+3,3*i+j]=diff[i]
            jac[-2,1:-2:3]=1/self.n;jac[-1,2:-2:3]=1/self.n
            for j in [-2,-1]:
                pert=v.copy();pert[j]+=h;jac[:,j]=(residual(pert)-initial)/h
            # Preserve the baseline free-boundary derivative path exactly.
            if fixed_y is not None or fixed_z is not None:
                jac[-2:,:]=0.
                if fixed_y is None:jac[-2,1:-2:3]=1/self.n
                else:jac[-2,-2]=1.
                if fixed_z is None:jac[-1,2:-2:3]=1/self.n
                else:jac[-1,-1]=1.
            return jac
        def stop_when_valid(v):
            if np.max(np.abs(residual(v)))<=tolerance:raise StopIteration
        options=dict(jac=jacobian,x_scale='jac',max_nfev=max_nfev,
                     ftol=1e-9,xtol=1e-9,gtol=1e-9,
                     callback=None if x==previous['position'][0] else stop_when_valid)
        linear_fallback=False
        try:
            solved=least_squares(residual,guess,**options)
        except np.linalg.LinAlgError:
            # The dense LAPACK SVD failed on some server runs. LSMR solves the
            # same trust-region least-squares problem without that factorization.
            # Final acceptance still uses the original contact/force residual.
            solved=least_squares(residual,guess,tr_solver='lsmr',**options)
            linear_fallback=True
        error,state=full(solved.x)
        state.update(residual=float(np.max(np.abs(error))),nfev=solved.nfev,
                     accepted=bool(np.max(np.abs(error))<=tolerance),reconfigured=False,
                     requested_preload_N=float(preload),linear_solver_fallback=linear_fallback)
        if recover and not state['accepted']:
            if not linear_fallback:
                # Try a different linear subproblem algorithm before looking for
                # another equilibrium. This recovered a stored stalled contact
                # step at the unchanged 1e-3 residual tolerance.
                alternative=least_squares(residual,guess,tr_solver='lsmr',**options)
                alternative_error,alternative_state=full(alternative.x)
                alternative_norm=float(np.max(np.abs(alternative_error)))
                if alternative_norm<=tolerance:
                    candidate=dict(state);candidate.update(alternative_state)
                    candidate.update(residual=alternative_norm,nfev=alternative.nfev,accepted=True,
                                     linear_solver_fallback=True,recovery_method='lsmr')
                    return candidate
            seed=self.energy_seed(x,preload,previous)
            if seed is not None:
                candidate=self.solve(x,preload,previous,guess=np.r_[seed[0].ravel()/force_scale,seed[1][1:]/length_scale])
                if candidate['accepted']:
                    candidate['reconfigured']=True
                    candidate['recovery_method']='energy'
                    return candidate
            from .trend_array import distribute_preload,vertical_stiffness
            # Search nearby free-Y equilibria only after ordinary continuation
            # has failed. Each candidate solves the SAME force/contact/history
            # equations. A discontinuous rearrangement is reported explicitly;
            # the unresolved transient must not be interpreted as a drag curve.
            for radius in [20e-6,100e-6,300e-6,1000e-6]:
                candidates=[]
                for dy in [-radius,radius]:
                    y=previous['position'][1]+dy
                    centers=self.guides+self.length[:,None]*self.axes
                    centers[:,0]+=x;centers[:,1]+=y
                    try:height,_=self.surface.query(centers[:,:2])
                    except ValueError:continue
                    z,loads=distribute_preload(height[None,:],vertical_stiffness(self.rods),preload)
                    seed=np.zeros_like(guess);seed[2:-2:3]=loads[0]/force_scale
                    seed[-2:]=[y/length_scale,z[0]/length_scale]
                    candidate=self.solve(x,preload,previous,guess=seed,tolerance=tolerance,max_nfev=max_nfev)
                    if candidate['accepted'] and np.all(candidate['compression']>=-1e-6) and np.all(candidate['compression']<self.length):
                        candidates.append(candidate)
                if candidates:
                    state=min(candidates,key=lambda c:np.linalg.norm(c['position'][1:]-previous['position'][1:]))
                    state['reconfigured']=True
                    state['recovery_method']='nearby_equilibrium'
                    break
        return state

    def energy_seed(self,x,preload,previous):
        """Dissipative energy minimization supplies a force-solver initial guess.

        Normal loads/directions are lagged only inside this initializer. Final
        acceptance always re-solves the original contact and friction residual.
        """
        if not self.rigid and np.any(self.spring):return None
        L=1e-5;n=self.n;d=1 if self.rigid else 4;m=n*d+2
        jc=np.zeros((n,3,m));jw=np.zeros_like(jc);H=np.zeros((m,m));v=np.zeros(m)
        jc[:,1,-2]=L;jc[:,2,-1]=L
        v[-2:]=previous['position'][1:]/L
        base=self.guides+self.length[:,None]*self.axes;base[:,0]+=x
        for i,rod in enumerate(self.rods):
            if self.rigid:
                jc[i,:,i]=-self.axes[i]*L
                if self.spring[i]==0:return None
                H[i,i]=L/(self.spring[i]*preload)
                v[i]=-np.dot(previous['force'][i],self.axes[i])*self.spring[i]/L
            else:
                e=rod.guide_rotation[:,1:]
                c=self.tables[i][1][-1];k=np.linalg.inv([[c[0],c[1]],[c[1],c[2]]])
                scale=np.array([L,L/self.length[i]])
                for j in range(2):
                    u=4*i+j;t=4*i+2+j
                    jc[i,:,u]=e[:,j]*L;jw[i,:,t]=np.cross(self.axes[i],e[:,j])*L/self.length[i]
                    H[np.ix_([u,t],[u,t])]=k*scale[:,None]*scale[None,:]/(preload*L)
                delta=previous['center'][i]-self.guides[i]-previous['position']-self.length[i]*self.axes[i]
                v[4*i:4*i+2]=e.T@delta/L
                v[4*i+2:4*i+4]=e.T@np.cross(previous['rotation'][i],self.axes[i])*self.length[i]/L
        normal=previous['normal'].copy();normal_load=np.maximum(0,np.sum(previous['force']*normal,axis=1))
        if normal_load.sum()<preload/2:
            height,_=self.surface.query((base+np.einsum('nki,i->nk',jc,v))[:,:2])
            active=height>=height.max()-1e-5
            normal_load=active*preload/max(1,active.sum())
        def geometry(w):
            center=base+np.einsum('nki,i->nk',jc,w)
            height,norm=self.surface.query(center[:,:2])
            grad=norm/norm[:,2,None]
            return center,(center[:,2]-self.radius-height)/L,grad,norm
        for outer in range(12):
            jmove=jc+np.cross(jw.transpose(0,2,1),-self.radius[:,None,None]*normal[:,None,:]).transpose(0,2,1)
            jmove-=normal[:,:,None]*np.einsum('nk,nki->ni',normal,jmove)[:,None,:]
            offset=base-previous['center']+np.cross(-previous['rotation'],-self.radius[:,None]*normal)
            offset-=normal*np.sum(offset*normal,axis=1)[:,None]
            mu=self.mu_kinetic+(self.mu_static-self.mu_kinetic)*np.exp(-np.sum((offset+np.einsum('nki,i->nk',jmove,v))**2,axis=1)/self.friction_transition_m**2)
            weight=mu*normal_load
            def objective(w):
                motion=offset+np.einsum('nki,i->nk',jmove,w)
                norm=np.sqrt(np.sum(motion*motion,axis=1)+1e-18)
                friction=weight[:,None]*motion/norm[:,None]
                gradient=H@w+np.einsum('nki,nk->i',jmove,friction)/(preload*L)
                gradient[-1]+=1
                return .5*w@H@w+w[-1]+np.dot(weight,norm)/(preload*L),gradient
            def gap(w):
                try:return geometry(w)[1]
                except ValueError:return np.full(n,-1e6)
            def gap_jac(w):
                return np.einsum('nk,nki->ni',geometry(w)[2],jc)/L
            try:
                result=minimize(objective,v,jac=True,method='SLSQP',
                    constraints={'type':'ineq','fun':gap,'jac':gap_jac},
                    options={'ftol':1e-10,'maxiter':120})
                center,g,grad,new_normal=geometry(result.x)
            except ValueError:return None
            if not hasattr(result,'multipliers'):return None
            v=result.x
            motion=offset+np.einsum('nki,i->nk',jmove,v)
            friction=weight[:,None]*motion/np.sqrt(np.sum(motion*motion,axis=1)+1e-18)[:,None]
            force=result.multipliers[:n,None]*preload*grad-friction
            actual=np.maximum(0,np.sum(force*new_normal,axis=1))
            if np.max(np.abs(actual-normal_load))<preload*1e-4 and np.min(g)>-1e-3:break
            normal_load=.5*normal_load+.5*actual;normal=new_normal
        return force,np.r_[x,v[-2:]*L]

    def initial(self,start_xy):
        centers=self.guides+self.length[:,None]*self.axes
        centers[:,:2]+=start_xy
        h,n=self.surface.query(centers[:,:2]);z=float(h.max())
        centers[:,2]+=z
        return dict(position=np.r_[start_xy,z],center=centers,rotation=np.zeros((self.n,3)),
                    force=np.zeros((self.n,3)),normal=n,slip=np.zeros(self.n,bool))

    def run(self,start_xy,preload,distance=.01,step=25e-6,progress=None,cpu_budget_s=45,preload_min_divisor=128,
            resume=None,state_observer=None):
        if resume is None:
            state=self.initial(start_xy);rows=[];attempts=0
        else:
            recoverable=resume['status'] in {'BALANCED_BUDGET_LIMIT','BALANCED_NUMERICAL_FAILURE'}
            recoverable|=(resume['status']=='BALANCED_AXIAL_RANGE' and
                          resume.get('reason')=='COMPRESSION_STOP_NOT_IMPLEMENTED')
            if not recoverable or resume.get('phase')!='drag':
                raise ValueError('Only numerical/budget or formerly unimplemented hard-stop cases can continue during drag')
            state={k:np.asarray(v) if isinstance(v,list) else v for k,v in resume['checkpoint'].items()}
            rows=[dict(row) for row in resume['rows']];attempts=resume.get('attempts',0)
        if state_observer:state_observer('initial' if resume is None else 'resume',state,None)
        started=time.process_time()
        last_position=state['position'].copy()
        def checkpoint():
            return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()}
        def failure(trial,phase):
            if state_observer:state_observer('failed_'+phase,trial,None)
            return dict(status='BALANCED_NUMERICAL_FAILURE',phase=phase,rows=rows,
                        failed_residual=trial['residual'],attempts=attempts,
                        attempted_preload_N=trial['requested_preload_N'],checkpoint=checkpoint(),
                        failed_state={k:trial[k].tolist() for k in ['position','force','normal','gap','compression']})
        def axial_range(trial,phase):
            s=trial['compression'];negative=np.flatnonzero(s < -1e-6)
            shortened=np.flatnonzero(s>=self.length)
            if not (len(negative) or len(shortened)):return None
            indices=negative if len(negative) else shortened
            if state_observer:state_observer('range_'+phase,trial,None)
            reason='AXIAL_EXTENSION_UNSUPPORTED' if len(negative) else 'EXPOSED_LENGTH_LIMIT'
            return dict(status='BALANCED_AXIAL_RANGE',phase=phase,reason=reason,rows=rows,attempts=attempts,
                        needle_indices=indices.tolist(),compression_m=s.tolist(),checkpoint=checkpoint())
        def record(phase):
            nonlocal last_position
            axial=np.sum(state['force']*self.axes,axis=1)
            stop_reaction=np.zeros(self.n)
            has_stop=np.isfinite(self.compression_stop)
            stop_reaction[has_stop]=np.maximum(0.,-axial[has_stop]-state['compression'][has_stop]/self.spring[has_stop])
            row=dict(phase=phase,x_m=float(state['position'][0]-start_xy[0]),
                     T_N=float(-state['force'][:,0].sum()),P_N=float(state['force'][:,2].sum()),
                     Y_m=float(state['position'][1]-start_xy[1]),Z_m=float(state['position'][2]),
                     residual=state['residual'],active=int(np.sum(state['normal_force']>preload/self.n*1e-3)),
                     slip=int(state['slip'].sum()),max_rotation_rad=float(np.linalg.norm(state['rotation'],axis=1).max()),
                     max_compression_m=float(state['compression'].max()),loads_N=state['force'][:,2].tolist(),
                     forces_N=state['force'].tolist(),normal_forces_N=state['normal_force'].tolist(),
                     compressions_m=state['compression'].tolist(),
                     compression_stop_reactions_N=stop_reaction.tolist(),
                     linear_solver_fallback=state.get('linear_solver_fallback',False),
                     reconfigured=state.get('reconfigured',False),
                     recovery_method=state.get('recovery_method','continuation'),
                     lateral_increment_m=float(state['position'][1]-last_position[1]),
                     unresolved_rearrangement=bool(abs(state['position'][1]-last_position[1])>2*step),
                     force_balance_error_N=float(np.hypot(state['force'][:,1].sum(),state['force'][:,2].sum()-requested_p)),
                     max_penetration_m=float(max(0.,-state['gap'].min())))
            if state_observer:state_observer('accepted',state,row)
            rows.append(row)
            last_position=state['position'].copy()
            if progress:progress(row)
        current_p=preload if resume is not None else 0.;dp=preload/8
        requested_p=current_p
        while current_p<preload-1e-12:
            if time.process_time()-started>cpu_budget_s:return dict(status='BALANCED_BUDGET_LIMIT',phase='preload',rows=rows,checkpoint=checkpoint())
            p=min(preload,current_p+dp)
            trial=self.solve(start_xy[0],p,state)
            if not trial['accepted']:
                if dp>preload/preload_min_divisor+1e-12:dp/=2;continue
                trial=self.solve(start_xy[0],p,state,recover=True)
                if not trial['accepted']:return failure(trial,'preload')
            limited=axial_range(trial,'preload')
            if limited:return limited
            state=trial;requested_p=p;record('preload');current_p=p;dp=min(preload/4,dp*1.5)
        target=start_xy[0]+distance;dx=step
        if resume is not None:
            if 'failed_state' in resume:
                dx=min(step,float(resume['failed_state']['position'][0]-state['position'][0]))
            elif len(rows)>1 and rows[-1]['phase']=='drag':
                dx=min(step,1.5*(rows[-1]['x_m']-rows[-2]['x_m']))
        while state['position'][0]<target-1e-12:
            if time.process_time()-started>cpu_budget_s:return dict(status='BALANCED_BUDGET_LIMIT',phase='drag',rows=rows,checkpoint=checkpoint())
            remaining=target-state['position'][0]
            actual_dx=min(dx,remaining);minimum_dx=min(step/8,remaining/8)
            x=state['position'][0]+actual_dx
            trial=self.solve(x,preload,state);attempts+=1
            if not trial['accepted']:
                if actual_dx>minimum_dx+1e-12:dx=max(minimum_dx,actual_dx/2);continue
                trial=self.solve(x,preload,state,recover=True)
                if not trial['accepted']:return failure(trial,'drag')
            limited=axial_range(trial,'drag')
            if limited:return limited
            state=trial;record('drag');dx=min(step,actual_dx*1.5)
        return dict(status='BALANCED_COMPLETE',rows=rows,attempts=attempts,
                    reconfigurations=sum(r['reconfigured'] for r in rows),checkpoint=checkpoint())
