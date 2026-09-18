"""Build the complete confirmation package using an existing portable runtime."""
import argparse,shutil,sys
from pathlib import Path


def build(target,runtime=None,codec_deps=None):
    if target.exists():raise FileExistsError(target)
    repo=Path(__file__).resolve().parents[1]
    ignore=shutil.ignore_patterns('__pycache__','*.pyc','__editable__*','*.pth')
    if runtime:
        shutil.copytree(runtime,target/'runtime',ignore=ignore)
    else:
        base=Path(sys.base_prefix);dest=target/'runtime';dest.mkdir(parents=True)
        for item in base.iterdir():
            if item.is_file() and item.suffix.lower() in ('.exe','.dll'):shutil.copy2(item,dest/item.name)
        shutil.copytree(base/'DLLs',dest/'DLLs',ignore=ignore)
        shutil.copytree(base/'Lib',dest/'Lib',ignore=shutil.ignore_patterns('site-packages','__pycache__','*.pyc'))
        shutil.copytree(base/'Library/bin',dest/'Library/bin')
        shutil.copytree(Path(sys.prefix)/'Lib/site-packages',dest/'Lib/site-packages',ignore=ignore)
        if codec_deps:
            for item in codec_deps.glob('orjson*'):shutil.copytree(item,dest/'Lib/site-packages'/item.name,ignore=ignore)
    for name in ('src','scripts','experiments'):
        shutil.copytree(repo/name,target/'program'/name,ignore=ignore)
    page=target/'program/monitor/dist';page.mkdir(parents=True)
    shutil.copy2(repo/'monitor/dist/confirmation.html',page/'confirmation.html')
    raw=Path('data/raw/mendeley_hcgcnm269w_v2');(target/'program'/raw).mkdir(parents=True)
    for name in ('P40.csv','P100.csv','P240.csv','source_metadata.json','ReadMe.txt'):
        shutil.copy2(repo/raw/name,target/'program'/raw/name)
    shutil.copy2(repo/'README.md',target/'README.md')
    shutil.copytree(repo/'docs',target/'docs')
    preamble='''@echo off
setlocal
cd /d "%~dp0"
set "PYTHONHOME="
set "PYTHONPATH="
set "PYTHONNOUSERSITE=1"
set "CUDA_PATH="
set "CUDA_HOME="
set "CUPY_CACHE_IN_MEMORY=1"
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PATH=%~dp0runtime;%~dp0runtime\\Library\\bin;%SystemRoot%\\System32;%SystemRoot%"
'''
    for batch,script in [('START_CONFIRMATION.bat','run_ijms_confirmation.py'),
                         ('TEST_ENVIRONMENT.bat','test_ijms_confirmation_environment.py')]:
        (target/batch).write_text(preamble+f'"%~dp0runtime\\python.exe" -I -B "%~dp0program\\scripts\\{script}" %*\npause\n',encoding='ascii')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',type=Path);p.add_argument('--codec-deps',type=Path)
    a=p.parse_args();build(a.output.resolve(),a.runtime.resolve() if a.runtime else None,a.codec_deps)
