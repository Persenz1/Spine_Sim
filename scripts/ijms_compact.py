"""Lossless numeric codec from the existing IJMS compact archive reader.

Supports old gzip/flat compact batches and new nested streaming batches.
"""
import gzip,hashlib,io,json,sqlite3,zipfile
from pathlib import Path
import numpy as np
try:
    import orjson
except ImportError:
    orjson=None

def loads(raw):
    if orjson is None:return json.loads(raw)
    try:return orjson.loads(raw)
    except orjson.JSONDecodeError:return json.loads(raw)

def canonical(j,mode):
    if mode=='stdlib' or orjson is None:return json.dumps(j,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=True).encode('utf-8')
    return orjson.dumps(j,option=orjson.OPT_SORT_KEYS)

def encode(raw,source,original_size):
    j=loads(raw);mode='stdlib' if orjson is None or b'NaN' in raw or b'Infinity' in raw else 'orjson'
    try:expected=hashlib.sha256(canonical(j,mode)).hexdigest()
    except TypeError:mode='stdlib';expected=hashlib.sha256(canonical(j,mode)).hexdigest()
    rows=j.get('rows',[]);arrays={}
    for field in dict.fromkeys(k for r in rows for k in r):
        if not all(field in r for r in rows):continue
        values=[r[field] for r in rows]
        try:
            arr=np.asarray(values)
            if arr.dtype.kind in 'biuf' and canonical(values,mode)==canonical(arr.tolist(),mode):arrays[field]=arr
        except (ValueError,TypeError,OverflowError):pass
    derived=[]
    if 'forces_N' in arrays and 'loads_N' in arrays:
        f=arrays['forces_N'][:,:,2];z=arrays['loads_N']
        if f.dtype==z.dtype and f.tobytes()==z.tobytes():
            del arrays['loads_N'];derived.append('loads_N')
    removed=set(arrays)|set(derived)
    content={k:v for k,v in j.items() if k!='rows'}
    if 'rows' in j:content['rows']=[{k:v for k,v in r.items() if k not in removed} for r in rows]
    schema={};payload=bytearray()
    for field,arr in arrays.items():
        data=np.frombuffer(arr.tobytes(),np.uint8).reshape(-1,arr.dtype.itemsize).T.copy().tobytes()
        schema[field]={'dtype':arr.dtype.str,'shape':arr.shape,'offset':len(payload),'length':len(data)};payload.extend(data)
    meta={'format':1,'source':source,'original_gz_bytes':original_size,'canonical_mode':mode,'canonical_sha256':expected,'arrays':schema,'derived':derived,'content':content}
    return json.dumps(meta,ensure_ascii=False,separators=(',',':'),allow_nan=True).encode(),payload

def decode(meta_bytes,payload,verify=False):
    meta=loads(meta_bytes);result=meta['content'];arrays={}
    if verify and meta['canonical_mode']=='orjson' and orjson is None:
        raise ImportError('orjson is required to verify legacy orjson canonical hashes')
    for field,desc in meta['arrays'].items():
        dtype=np.dtype(desc['dtype']);raw=memoryview(payload)[desc['offset']:desc['offset']+desc['length']]
        a=np.frombuffer(raw,np.uint8).reshape(dtype.itemsize,-1).T.copy().reshape(-1).view(dtype).reshape(desc['shape']);arrays[field]=a
        for row,v in zip(result['rows'],a.tolist()):row[field]=v
    if 'loads_N' in meta['derived']:
        for row,v in zip(result['rows'],arrays['forces_N'][:,:,2].tolist()):row['loads_N']=v
    if verify and hashlib.sha256(canonical(result,meta['canonical_mode'])).hexdigest()!=meta['canonical_sha256']:
        raise ValueError('Restored values/types differ: '+meta['source'])
    return result,meta


def pack_trace(result,source):
    raw=json.dumps(result,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()
    meta,payload=encode(raw,source,0)
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('0.json',meta);archive.writestr('0.bin',payload)
    return output.getvalue()


def unpack_trace(data,verify=False):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return decode(archive.read('0.json'),archive.read('0.bin'),verify=verify)[0]


class TraceReader:
    def __init__(self,root):
        self.root=Path(root);self.db=None;self.zip=None;self.zip_name=None

    def read(self,relative,verify=False):
        source=self.root/relative
        if source.exists():
            with gzip.open(source,'rb') as f:return loads(f.read())
        if self.db is None:
            self.db=sqlite3.connect((self.root/'compact/index.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
        row=self.db.execute('SELECT archive,member FROM traces WHERE source=?',(relative.replace('\\','/'),)).fetchone()
        if row is None:raise FileNotFoundError(relative)
        name,member=row
        if name!=self.zip_name:
            if self.zip:self.zip.close()
            self.zip=zipfile.ZipFile(self.root/'compact'/name);self.zip_name=name
        if f'{member}.zip' in self.zip.namelist():
            return unpack_trace(self.zip.read(f'{member}.zip'),verify=verify)
        return decode(self.zip.read(f'{member}.json'),self.zip.read(f'{member}.bin'),verify=verify)[0]

    def close(self):
        if self.zip:self.zip.close()
        if self.db:self.db.close()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
