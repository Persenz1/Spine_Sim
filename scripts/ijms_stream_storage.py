"""A single sequential batch writer. Final ZIPs are recoverable before indexing."""
import io,json,os,sqlite3,time,zipfile
from pathlib import Path


class BatchWriter:
    def __init__(self,root,max_bytes=64*2**20,max_cases=32,flush_seconds=30):
        self.root=Path(root);self.dest=self.root/'compact';self.dest.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.dest/'index.sqlite3')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS cases(case_key TEXT PRIMARY KEY,status TEXT NOT NULL,summary_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS traces(source TEXT PRIMARY KEY,archive TEXT,member INTEGER);
            CREATE TABLE IF NOT EXISTS batches(archive TEXT PRIMARY KEY,cases INTEGER);
        ''')
        self.max_bytes=max_bytes;self.max_cases=max_cases;self.flush_seconds=flush_seconds
        self.entries=[];self.size=0;self.last_flush=time.monotonic()
        known={r[0] for r in self.db.execute('SELECT archive FROM batches')}
        for path in sorted(self.dest.glob('block-*.zip')):
            if path.name not in known:
                with zipfile.ZipFile(path) as archive:records=json.loads(archive.read('summaries.json'))
                self._index(path.name,records)
        self.sequence=max([int(p.stem.split('-')[1])+1 for p in self.dest.glob('block-*.zip')]+[0])

    def _index(self,name,records):
        with self.db:
            for i,record in enumerate(records):
                row=record['summary'];key=record['key']
                self.db.execute('INSERT INTO cases VALUES(?,?,?)',(key,row['status'],json.dumps(row,ensure_ascii=False)))
                if record['has_trace']:self.db.execute('INSERT INTO traces VALUES(?,?,?)',(key,name,i))
            self.db.execute('INSERT INTO batches VALUES(?,?)',(name,len(records)))

    def add(self,key,summary,trace):
        if self.entries and self.size+len(trace or b'')>self.max_bytes:self.flush()
        self.entries.append((key,summary,trace));self.size+=len(trace or b'')
        if self.size>=self.max_bytes or len(self.entries)>=self.max_cases:self.flush()

    def flush_due(self):
        if time.monotonic()-self.last_flush>=self.flush_seconds:self.flush()

    def flush(self):
        if not self.entries:return
        name=f'block-{self.sequence:06d}.zip';path=self.dest/name
        records=[dict(key=key,summary=row,has_trace=trace is not None) for key,row,trace in self.entries]
        # Inner case ZIPs are already compressed by CPU workers in memory.
        # ZIP_STORED avoids recompression and makes the HDD write sequential.
        with path.with_suffix('.partial').open('wb') as file:
            with zipfile.ZipFile(file,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                for i,(_,_,trace) in enumerate(self.entries):
                    if trace is not None:archive.writestr(f'{i}.zip',trace)
                archive.writestr('summaries.json',json.dumps(records,ensure_ascii=False).encode())
            file.flush();os.fsync(file.fileno())
        os.replace(path.with_suffix('.partial'),path)
        self._index(name,records)
        self.sequence+=1;self.entries=[];self.size=0;self.last_flush=time.monotonic()

    def close(self):
        try:self.flush()
        finally:self.db.close()
