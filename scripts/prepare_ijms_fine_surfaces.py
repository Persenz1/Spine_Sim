"""Generate frozen fine-screen surfaces on CUDA, stage on SSD, publish sequentially."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import argparse, gc, json, shutil, time
from pathlib import Path
import numpy as np
from spine_sim.terrain import generate_terrain

MATERIALS=[('concrete','rough_wall'),('red_brick','fired_brick_standard'),('sandpaper','P40'),('sandpaper','P100'),('sandpaper','P240')]
DOMAINS={'small':(.068,.058,[-.029,-.029],100,2026091000), 'large':(.140,.130,[-.065,-.065],20,2026091300)}
SUFFIX='-r100-envelope20um-sigma10um.npy'

def envelope(raw,target):
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    radius=100e-6; dx=20e-6
    h=cp.asarray(np.load(raw,mmap_mode='r')[::2,::2],dtype=cp.float64); out=cp.full_like(h,-cp.inf)
    for j in range(-int(radius/dx),int(radius/dx)+1):
        for i in range(-int(radius/dx),int(radius/dx)+1):
            r2=radius**2-(i*dx)**2-(j*dx)**2
            if r2<0:continue
            y0,y1=max(0,-j),min(h.shape[0],h.shape[0]-j)
            x0,x1=max(0,-i),min(h.shape[1],h.shape[1]-i)
            cp.maximum(out[y0:y1,x0:x1],h[y0+j:y1+j,x0+i:x1+i]+np.sqrt(r2)-radius,out=out[y0:y1,x0:x1])
    out=gaussian_filter(out,(.5,.5))
    np.save(target,cp.asnumpy(out))

def build_surface(job):
    """Independent CUDA producer; publication to HDD happens only in the parent."""
    domain,sample,material,subtype,scratch=job
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('CUDA required; no CPU fallback')
    os.environ['CUPY_CACHE_DIR']=str(Path(scratch)/'cuda-cache')
    sx,sy,origin,count,seed0=DOMAINS[domain]
    stem=f'{material}-{subtype}-s{sample:03d}'
    source=Path(scratch)/domain/stem;source.mkdir(parents=True,exist_ok=True)
    names=[stem+'.npy',stem+SUFFIX,stem+'.json'];tick=time.monotonic()
    if all((source/n).is_file() for n in names):
        return dict(source=str(source),domain=domain,stem=stem,names=names,generation_s=0,prepare_s=0)
    terrain=generate_terrain(material=material,subtype=subtype,seed=seed0+sample,mode='synthetic',backend='cuda',size_x_m=sx,size_y_m=sy,resolution_m=1e-5)
    generation_s=time.monotonic()-tick
    raw=source/names[0]
    np.save(raw,terrain.height)
    meta=dict(terrain.metadata)
    meta.update(measurement_probe=terrain.measurement_probe,measurement_tolerance_m=terrain.measurement_tolerance_m,
        array_storage=dict(dtype='float32',source_dtype='float32',origin_xy_m=origin),
        sample_scope='synthetic_realization; measured source specimens are not independent per seed',
        fine_surface_set=domain,sample_index=sample,
        actual_height_statistics=dict(mean_m=float(np.mean(terrain.height,dtype=np.float64)),rms_about_mean_m=float(np.std(terrain.height,dtype=np.float64)),minimum_m=float(terrain.height.min()),maximum_m=float(terrain.height.max())))
    del terrain;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    envelope(raw,source/names[1])
    (source/names[2]).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return dict(source=str(source),domain=domain,stem=stem,names=names,generation_s=generation_s,prepare_s=time.monotonic()-tick)

def main():
    from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
    import multiprocessing
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True)
    p.add_argument('--coarse-surfaces',type=Path)
    p.add_argument('--domain',choices=['all','small','large'],default='all')
    p.add_argument('--workers',type=int,default=3);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.scratch.mkdir(parents=True,exist_ok=True)
    os.environ['CUPY_CACHE_DIR']=str(a.scratch/'cuda-cache')
    root=a.package/'data'/'surfaces';root.mkdir(parents=True,exist_ok=True)
    status=a.package/'data'/'generation_status.json';start=time.time();done=0;new=0;queues={}
    domains={k:v for k,v in DOMAINS.items() if a.domain=='all' or k==a.domain}
    total=sum(v[3]*len(MATERIALS) for v in domains.values())
    for domain,(sx,sy,origin,count,seed0) in domains.items():
        dest=root/domain;dest.mkdir(exist_ok=True);queues[domain]=[]
        for sample in range(count):
            for material,subtype in MATERIALS:
                stem=f'{material}-{subtype}-s{sample:03d}';names=[stem+'.npy',stem+SUFFIX,stem+'.json']
                if all((dest/n).is_file() for n in names):
                    for n,step in zip(names,[1e-5,2e-5]):
                        arr=np.load(dest/n,mmap_mode='r')
                        allowed=(np.dtype('float32'),np.dtype('float64')) if step==1e-5 else (np.dtype('float64'),)
                        if arr.shape!=(round(sy/step)+1,round(sx/step)+1) or arr.dtype not in allowed:raise ValueError(f'Invalid existing terrain: {dest/n}')
                        del arr
                    done+=1
                elif domain=='small' and sample<4:
                    if a.coarse_surfaces is None:raise ValueError('--coarse-surfaces is required to reuse original small inputs')
                    for name in names:
                        tmp=dest/(name+'.partial');shutil.copyfile(a.coarse_surfaces/name,tmp);os.replace(tmp,dest/name)
                    done+=1
                else:queues[domain].append((domain,sample,material,subtype,str(a.scratch)))
    initial=done
    print(f'Resume: {done}/{total}; CUDA producers={a.workers} small, 1 large; one sequential writer',flush=True)
    import queue,threading
    from concurrent.futures import as_completed
    ready_queue=queue.Queue()
    produced=0
    def produce():
        nonlocal produced
        try:
            for domain,jobs in queues.items():
                count=a.workers if domain=='small' else 1
                if a.limit:jobs=jobs[:max(0,a.limit-produced)]
                with ProcessPoolExecutor(max_workers=count,mp_context=multiprocessing.get_context('spawn')) as pool:
                    futures=[pool.submit(build_surface,j) for j in jobs]
                    for future in as_completed(futures):
                        record=future.result();produced+=1
                        ready_queue.put(record)
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            ready_queue.put(exc)
        finally:ready_queue.put(None)
    # The entire remaining frozen input fits on the SSD; GPU production never
    # waits for HDD writes. Parent publishes one file at a time and frees staging.
    expected=sum((round(DOMAINS[d][0]/1e-5)+1)*(round(DOMAINS[d][1]/1e-5)+1)*6*len(j) for d,j in queues.items())
    if shutil.disk_usage(a.scratch).free<expected+20*2**30:raise RuntimeError('Insufficient SSD space for independent production pipeline')
    producer=threading.Thread(target=produce);producer.start()
    while True:
        record=ready_queue.get()
        if record is None:break
        if isinstance(record,BaseException):raise record
        domain=record['domain'];source=Path(record['source']);dest=root/domain;tick=time.monotonic()
        for name in record['names']:
            tmp=dest/(name+'.partial')
            if source.anchor.lower()==dest.anchor.lower():os.replace(source/name,tmp)
            else:shutil.copyfile(source/name,tmp)
            os.replace(tmp,dest/name)
        for name in record['names']:
            if (source/name).exists():(source/name).unlink()
        source.rmdir();done+=1;new+=1
        state=dict(status='COMPLETED' if done==total else 'GENERATING',completed_surfaces=done,total_surfaces=total,
            current=f"{domain}/{record['stem']}",elapsed_s=time.time()-start,new_this_run=new,initial_surfaces=initial,
            produced_this_run=produced,ready_on_ssd=produced-new,
            generation_s=record['generation_s'],prepare_s=record['prepare_s'],copy_s=time.monotonic()-tick)
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(state),encoding='utf-8');os.replace(temp,status)
        print(json.dumps(state),flush=True)
    producer.join()

if __name__=='__main__':main()
