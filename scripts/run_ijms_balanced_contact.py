"""Measured intermediate-fidelity pilots; never starts the 336000-case queue."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,time,gzip
from functools import lru_cache
from pathlib import Path
import numpy as np
from spine_sim.small_array_campaign import read_config,design_table,prepare_surface,Shard
from spine_sim.ijms import build_rods
from spine_sim.balanced_contact import CondensedArray,EnvelopeSurface,MODEL_VERSION
from spine_sim.placement_retry import run_with_placements


@lru_cache(maxsize=4)
def campaign_inputs(config_path):
    settings=read_config(config_path)
    os.environ['CUPY_CACHE_DIR']=str(Path(settings['output_dir'])/'tmp'/'cuda-cache')
    import spine_sim.small_array_campaign as campaign_module
    campaign_module.TEMP_ROOT=Path(settings['output_dir'])/'tmp'
    base=read_config(Path(config_path).parent/settings['base_config'])
    base['design_mode']=settings.get('design_mode','prescribed')
    return settings,base,{d['name']:d for d in design_table(base)}


@lru_cache(maxsize=2)
def cached_envelope(path,dx,dy,origin):
    return EnvelopeSurface(np.load(path,mmap_mode='r'),dx,dy,origin)


def result_name(design,material,preload,rigid,distance_mm,step_um,sample_index=None,compressed=False):
    tag=f"{design}-{material}-P{preload:g}-{'rigid' if rigid else 'elastic'}-{distance_mm:g}mm-{step_um:g}um-v{MODEL_VERSION.rsplit('-',1)[-1]}"
    if sample_index is not None:tag=f's{sample_index:03d}/{material}/{tag}'
    return tag+'.json'+('.gz' if compressed else '')


def read_result(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',encoding='utf-8') as stream:return json.load(stream)


def prepare_envelope(surface,radius,root,grid_um=20,sigma_um=10):
    target=root/'surfaces'/f"{Path(surface['path']).stem}-r{radius*1e6:.0f}-envelope{grid_um:g}um-sigma{sigma_um:g}um.npy"
    target.parent.mkdir(parents=True,exist_ok=True)
    sx=int(round(grid_um*1e-6/surface['dx_m']));sy=int(round(grid_um*1e-6/surface['dy_m']))
    dx=surface['dx_m']*sx;dy=surface['dy_m']*sy
    if not target.exists():
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter
        h=cp.asarray(np.load(surface['path'],mmap_mode='r')[::sy,::sx])
        out=cp.full_like(h,-cp.inf)
        for j in range(-int(radius/dy),int(radius/dy)+1):
            for i in range(-int(radius/dx),int(radius/dx)+1):
                r2=radius**2-(i*dx)**2-(j*dy)**2
                if r2<0:continue
                y0,y1=max(0,-j),min(h.shape[0],h.shape[0]-j)
                x0,x1=max(0,-i),min(h.shape[1],h.shape[1]-i)
                cp.maximum(out[y0:y1,x0:x1],h[y0+j:y1+j,x0+i:x1+i]+np.sqrt(r2)-radius,out=out[y0:y1,x0:x1])
        out=gaussian_filter(out,(sigma_um*1e-6/dy,sigma_um*1e-6/dx))
        np.save(target,cp.asnumpy(out))
    return cached_envelope(str(target),dx,dy,tuple(surface['origin_xy_m']))


def main(arguments=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('experiments/ijms_balanced_contact.json'))
    p.add_argument('--design',default='r50_a70_spring800_4x4_p5')
    p.add_argument('--material',default='P240');p.add_argument('--preload',type=float,default=.5)
    p.add_argument('--sample-index',type=int)
    p.add_argument('--distance-mm',type=float,default=10);p.add_argument('--step-um',type=float)
    p.add_argument('--rigid',action='store_true')
    p.add_argument('--needle-model',choices=['auto','rigid','elastic'],default='auto',
                   help='auto: spring-mounted needles rigid, fixed-mounted needles elastic')
    args=p.parse_args(arguments);started=time.perf_counter()
    saved_settings,base,designs=campaign_inputs(str(args.config.resolve()))
    settings=dict(saved_settings)
    if args.sample_index is not None:settings['sample_index']=args.sample_index
    if args.step_um is None:args.step_um=settings['path_step_um']
    root=Path(settings['output_dir'])
    for d in ['results','logs','tmp','surfaces']:(root/d).mkdir(parents=True,exist_ok=True)
    os.environ['TEMP']=os.environ['TMP']=str(root/'tmp')
    design=designs[args.design]
    mi=next(i for i,m in enumerate(base['materials']) if m['subtype']==args.material)
    surface=prepare_surface(Path(settings.get('surface_root','E:/TestData/IJMS/trend')),base,Shard(settings['sample_index'],mi,0,True,()),terrain_backend='cuda')
    field=prepare_envelope(surface,design['array']['spine']['tip_radius_m'],Path(settings.get('envelope_root',root)),
                           settings['surface_envelope_grid_um'],settings['surface_smoothing_sigma_um'])
    rigid=args.rigid or args.needle_model=='rigid' or (args.needle_model=='auto' and settings['needle_model'][design['array']['spine']['mount_type']]=='rigid')
    array=CondensedArray(build_rods(design['array']),field,rigid=rigid,
                         mu_static=base['solver']['friction_static'],mu_kinetic=base['solver']['friction_kinetic'],
                         friction_transition_m=settings['friction_transition_um']*1e-6,
                         tolerance=settings['relative_solver_tolerance'])
    preparation=time.perf_counter()-started;solve_start=time.perf_counter()
    name=result_name(args.design,args.material,args.preload,rigid,args.distance_mm,args.step_um,
                     settings['sample_index'] if settings.get('sample_indices') is not None else None,
                     settings.get('compress_results',False))
    log_path=root/'logs'/(name+'.jsonl')
    log=None
    if settings.get('step_logs',True):
        log_path.parent.mkdir(parents=True,exist_ok=True)
        log=log_path.open('w',encoding='utf-8',buffering=1)
    def progress(row):
        if log:log.write(json.dumps(dict(row,elapsed_s=time.perf_counter()-solve_start))+'\n')
    budget=settings['per_case_cpu_budget_s'] if rigid else settings['fixed_case_cpu_budget_s']
    relocation=settings.get('preload_relocation',{})
    offsets=relocation['offsets_m'] if relocation.get('enabled',False) else [[0.,0.]]
    result=run_with_placements(array,np.array(base['path']['start_xy_m'])+base['surface']['placement_xy_m'],
                     args.preload,args.distance_mm/1000,args.step_um/1e6,progress=progress,
                     offsets_m=offsets,attempt_cpu_budget_s=budget,
                     total_cpu_budget_s=relocation.get('total_cpu_budget_s',budget),
                     preload_min_divisor=settings.get('preload_min_divisor',128))
    if log:log.close()
    configuration=dict(vars(args),config=str(args.config))
    result.update(model=MODEL_VERSION,configuration=configuration,numerical_settings=settings,
                  preparation_s=preparation,solve_s=time.perf_counter()-solve_start,
                  approximations=['small_rotation_linear_beam' if not rigid else 'rigid_needle',
                    'spring_axial_force_without_configurational_term',f"smooth_single_support_envelope_{settings['surface_envelope_grid_um']:g}um",
                    'no_body_or_cap_checks',f"static_kinetic_transition_{settings['friction_transition_um']:g}um",
                    'reported_quasistatic_rearrangements_without_transient_resolution'],rigid=rigid)
    target=root/'results'/name;target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(target.suffix+'.tmp')
    if settings.get('compress_results',False):
        with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=1) as stream:json.dump(result,stream,ensure_ascii=False,allow_nan=False)
    else:temporary.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temporary,target)
    if settings.get('step_logs',True):
        print(json.dumps({k:v for k,v in result.items() if k not in ['rows','failed_state','checkpoint','numerical_settings','placement_attempts']}),flush=True)
    return dict(result,trace_file=name)

if __name__=='__main__':main()
