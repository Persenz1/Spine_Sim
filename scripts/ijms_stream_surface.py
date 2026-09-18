"""One surface owner, read-only shared spline coefficients for spawned workers."""
import gc
import hashlib
import mmap
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
from spine_sim.balanced_contact import EnvelopeSurface


class SharedSurface:
    def __init__(self,surface):
        self.blocks=[]
        self.descriptor=dict(x=surface.x,y=surface.y,parts={})
        try:
            for name in ('spline','slope_x','slope_y'):
                spline=getattr(surface,name);tx,ty,c=spline.tck
                block=SharedMemory(create=True,size=c.nbytes);self.blocks.append(block)
                np.ndarray(c.shape,dtype=c.dtype,buffer=block.buf)[:]=c
                self.descriptor['parts'][name]=dict(name=block.name,shape=c.shape,dtype=c.dtype.str,
                                                    tx=tx,ty=ty,degrees=spline.degrees)
        except BaseException:
            self.close();raise

    def close(self):
        for block in self.blocks:
            block.close();block.unlink()
        self.blocks=[]


def attach_surface(descriptor):
    result=EnvelopeSurface.__new__(EnvelopeSurface)
    result.x=descriptor['x'];result.y=descriptor['y'];blocks=[]
    for name,part in descriptor['parts'].items():
        # FILE_MAP_COPY: reads share physical pages, accidental writes are
        # private. NumPy exposes a writable view so FITPACK does not make a
        # complete 364 MB copy per query. The evaluator itself never writes.
        size=int(np.prod(part['shape']))*np.dtype(part['dtype']).itemsize
        block=mmap.mmap(-1,size,tagname=part['name'],access=mmap.ACCESS_COPY);blocks.append(block)
        coeff=np.ndarray(part['shape'],dtype=part['dtype'],buffer=block)
        setattr(result,name,RectBivariateSpline._from_tck((part['tx'],part['ty'],coeff,*part['degrees'])))
    return result,blocks

