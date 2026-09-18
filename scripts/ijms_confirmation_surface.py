"""CUDA generation, bounded shared terrain memory, and replay metadata."""
import gc,hashlib,json,platform,sys,time,zipfile
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
import numpy as np
from spine_sim.balanced_contact import EnvelopeSurface
from ijms_confirmation import SURFACE,PROGRAM
from ijms_stream_surface import SharedSurface


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def envelope(height):
    # Operation-for-operation copy of the baseline CUDA envelope; no file I/O.
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter
    radius=100e-6;dx=20e-6
    h=cp.asarray(height[::2,::2],dtype=cp.float64);out=cp.full_like(h,-cp.inf)
    for j in range(-int(radius/dx),int(radius/dx)+1):
        for i in range(-int(radius/dx),int(radius/dx)+1):
            r2=radius**2-(i*dx)**2-(j*dx)**2
            if r2<0:continue
            y0,y1=max(0,-j),min(h.shape[0],h.shape[0]-j)
            x0,x1=max(0,-i),min(h.shape[1],h.shape[1]-i)
            cp.maximum(out[y0:y1,x0:x1],h[y0+j:y1+j,x0+i:x1+i]+np.sqrt(r2)-radius,out=out[y0:y1,x0:x1])
    return cp.asnumpy(gaussian_filter(out,(.5,.5)))


def environment():
    import cupy as cp,scipy
    props=cp.cuda.runtime.getDeviceProperties(0)
    name=props['name'];name=name.decode() if isinstance(name,bytes) else name
    return dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,cupy=cp.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),cuda_driver=cp.cuda.runtime.driverGetVersion(),
        device=name,compute_capability=[props['major'],props['minor']],platform=platform.platform())


def generate(job,size=None):
    import cupy as cp
    from spine_sim.terrain import generate_terrain
    if cp.cuda.runtime.getDeviceCount()<1:raise RuntimeError('CUDA GPU required; no CPU fallback')
    sx,sy=size or (SURFACE['size_x_m'],SURFACE['size_y_m'])
    params=dict(material=job['material_family'],subtype=job['material'],seed=job['seed'],
                mode='synthetic',backend='cuda',size_x_m=sx,size_y_m=sy,resolution_m=1e-5)
    terrain=generate_terrain(**params)
    raw_hash=digest(terrain.height);height=envelope(terrain.height)
    meta=dict(generation_parameters=params,terrain_metadata=terrain.metadata,
        raw_sha256=raw_hash,raw_dtype=str(terrain.height.dtype),raw_shape=list(terrain.height.shape),
        envelope_sha256=digest(height),envelope_dtype=str(height.dtype),envelope_shape=list(height.shape),
        processing=SURFACE,environment=environment(),surface_id=job['surface_id'],sample_index=job['sample'])
    del terrain;gc.collect();cp.get_default_memory_pool().free_all_blocks();cp.get_default_pinned_memory_pool().free_all_blocks()
    return height,meta


def prepare(job):
    started=time.perf_counter();height,meta=generate(job)
    # Verify the full production shape for the first realization of each material.
    if job['sample']==0:
        original=(meta['raw_sha256'],meta['envelope_sha256'])
        del height
        height,again=generate(job)
        if original!=(again['raw_sha256'],again['envelope_sha256']):
            raise ValueError('Terrain regeneration is not identical: '+job['surface_id'])
        meta['regeneration_verified']=True
    surface=EnvelopeSurface(height,20e-6,20e-6,SURFACE['origin_xy_m'])
    del height
    owner=SharedSurface(surface);meta['prepare_s']=time.perf_counter()-started
    return owner,meta


def snapshot(results):
    """Code and actual measured source patches accompany results, once per batch."""
    dest=Path(results)/'reproduction';dest.mkdir(parents=True,exist_ok=True)
    sources=[]
    for name in ('src','scripts','experiments'):
        sources.extend(p for p in (PROGRAM/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc')
    raw=PROGRAM/'data/raw/mendeley_hcgcnm269w_v2'
    sources.extend(raw/name for name in ('P40.csv','P100.csv','P240.csv','source_metadata.json','ReadMe.txt'))
    identity=hashlib.sha256()
    for p in sorted(sources):
        identity.update(p.relative_to(PROGRAM).as_posix().encode());identity.update(p.read_bytes())
    value=identity.hexdigest();marker=dest/'source_identity.json'
    if marker.exists():
        if json.loads(marker.read_text('utf-8'))['sha256']!=value:raise ValueError('Code/source data changed; use a new output directory')
        return
    with zipfile.ZipFile(dest/'source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for p in sources:archive.write(p,p.relative_to(PROGRAM).as_posix())
    marker.write_text(json.dumps(dict(sha256=value,environment=environment()),ensure_ascii=False,indent=2),'utf-8')


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Reconstruct a result terrain and verify its raw/envelope identity')
    parser.add_argument('--metadata',type=Path,required=True);args=parser.parse_args()
    meta=json.loads(args.metadata.read_text('utf-8'));p=meta['generation_parameters']
    job=dict(material_family=p['material'],material=p['subtype'],seed=p['seed'],surface_id=meta['surface_id'],sample=meta['sample_index'])
    _,actual=generate(job,(p['size_x_m'],p['size_y_m']))
    for field in ('raw_sha256','envelope_sha256'):
        if meta[field]!=actual[field]:raise ValueError('Regenerated data differ: '+field)
    print('Raw terrain and effective envelope reproduce exactly; no terrain files written.')


if __name__=='__main__':main()
