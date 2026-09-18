"""Trend-only normal load sharing and roughness resistance.

This deliberately reduced model is NOT the finite-rod/contact solver. It uses
unloaded linear vertical compliance, nominal XY tracks, a sampled sphere-height
envelope, and continuous sliding. It omits 3-D drift, stick-slip anchoring,
finite retraction stiffening and body collisions. No nonlinear iterations occur.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

MODEL_VERSION = 'ijms-trend-normal-envelope-1'


@lru_cache(maxsize=128)
def section_coefficients(parameters):
    """Euler-Bernoulli end-force compliance of the actual tapered section."""
    p=parameters
    nodes,weights=np.polynomial.legendre.leggauss(48)
    cuts=sorted(set([0.,min(p.taper_length_m,p.free_length_m),p.free_length_m]))
    compliance=0.
    for low,high in zip(cuts[:-1],cuts[1:]):
        if high==low:
            continue
        u=(low+high)/2+(high-low)/2*nodes
        compliance+=(high-low)/2*np.sum(weights*u*u/(p.young_modulus_Pa*p.section_second_moment_m4(u)))
    u=np.linspace(0.,p.free_length_m,256)
    bending_factor=float(np.max(u*p.section_radius_m(u)/p.section_second_moment_m4(u)))
    axial_factor=float(np.max(1/p.section_area_m2(u)))
    return float(compliance),bending_factor,axial_factor


def vertical_stiffness(rods):
    values=[]
    for rod in rods:
        p=rod.parameters
        bending,_,_=section_coefficients(p)
        axial=rod.axis[2]**2/p.spring_stiffness_N_per_m if p.mount_type=='spring' else 0.
        compliance=axial+(1-rod.axis[2]**2)*bending
        if compliance<=0:
            raise ValueError('trend vertical compliance is zero for this assembly')
        values.append(1/compliance)
    return np.asarray(values)


def sampled_envelope(height, dx, dy, origin, centers_xy, radius, spacing, *, chunk=1024):
    """Sample max[h(x+u,y+v)+sqrt(r²-u²-v²)]-r, using bilinear h.

    The stencil is a declared coarse geometry approximation, not an exact
    triangle/sphere contact query. Only the required tracks are sampled.
    """
    count=max(1,int(np.ceil(radius/spacing)))
    grid=np.linspace(-radius,radius,2*count+1)
    ox,oy=np.meshgrid(grid,grid)
    mask=ox*ox+oy*oy<=radius*radius*(1+1e-12)
    ox,oy=ox[mask],oy[mask]
    cap=np.sqrt(np.maximum(0.,radius*radius-ox*ox-oy*oy))-radius
    centers=np.asarray(centers_xy).reshape(-1,2)
    result=np.empty(len(centers))
    for begin in range(0,len(centers),chunk):
        points=centers[begin:begin+chunk]
        u=(points[:,0,None]+ox-origin[0])/dx
        v=(points[:,1,None]+oy-origin[1])/dy
        i,j=np.floor(u).astype(np.intp),np.floor(v).astype(np.intp)
        if i.min()<0 or j.min()<0 or i.max()>=height.shape[1]-1 or j.max()>=height.shape[0]-1:
            raise ValueError('trend track exceeds the supplied surface domain')
        a,b=u-i,v-j
        z=((1-a)*(1-b)*height[j,i]+a*(1-b)*height[j,i+1]
           +(1-a)*b*height[j+1,i]+a*b*height[j+1,i+1])
        if not np.all(np.isfinite(z)):
            raise ValueError('invalid height sample on a trend track')
        result[begin:begin+len(points)]=np.max(z+cap,axis=1)
    return result.reshape(np.asarray(centers_xy).shape[:-1])


def distribute_preload(heights, stiffness, preload):
    """Exact active-set water filling for Pi=ki*max(hi-Z,0), sum Pi=P."""
    h=np.asarray(heights,float)
    k=np.broadcast_to(stiffness,h.shape)
    order=np.argsort(-h,axis=1)
    hs=np.take_along_axis(h,order,axis=1)
    ks=np.take_along_axis(k,order,axis=1)
    levels=(np.cumsum(ks*hs,axis=1)-preload)/np.cumsum(ks,axis=1)
    next_height=np.concatenate((hs[:,1:],np.full((len(h),1),-np.inf)),axis=1)
    active_count=np.argmax(levels>=next_height,axis=1)
    z=levels[np.arange(len(h)),active_count]
    loads=k*np.maximum(h-z[:,None],0.)
    return z,loads


def solve_trend(x, envelope, rods, preload, mu):
    k=vertical_stiffness(rods)
    z,loads=distribute_preload(envelope,k,preload)
    slope=np.gradient(envelope,x,axis=0,edge_order=2)
    resistance_per_rod=loads*(mu+slope)
    resistance=resistance_per_rod.sum(axis=1)
    compression=np.zeros_like(loads)
    stress=np.zeros_like(loads)
    invalid=np.zeros(len(x),dtype=bool)
    for i,rod in enumerate(rods):
        p=rod.parameters
        if p.mount_type=='spring':
            compression[:,i]=loads[:,i]*(-rod.axis[2])/p.spring_stiffness_N_per_m
            invalid|=compression[:,i]>=p.compression_limit_m
        force=np.column_stack((-resistance_per_rod[:,i],np.zeros(len(x)),loads[:,i]))
        axial=force@rod.axis
        transverse=np.linalg.norm(force-axial[:,None]*rod.axis,axis=1)
        _,bending_factor,axial_factor=section_coefficients(p)
        stress[:,i]=np.abs(axial)*axial_factor+transverse*bending_factor
        if p.allowable_stress_Pa is not None:
            invalid|=stress[:,i]>p.allowable_stress_Pa
    return dict(T_N=resistance,Z_m=z,loads_N=loads,
                active_count=np.sum(loads>max(1e-12,preload*1e-8),axis=1),
                normal_energy_J=np.sum(loads*loads/(2*k),axis=1),
                max_compression_m=compression.max(axis=1),max_stress_estimate_Pa=stress.max(axis=1),
                range_flag=invalid,normal_balance_error_N=float(np.max(np.abs(loads.sum(axis=1)-preload))))
