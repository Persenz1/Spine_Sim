"""Persistent wall-clock waves; starting a new wave is always explicit."""
import time


def begin_wave(connection,hours,workers,completed,*,start_next=False,now=None):
    if hours<=0:raise ValueError('wave hours must be positive')
    now=time.time() if now is None else now
    connection.execute('''CREATE TABLE IF NOT EXISTS waves(
        wave_id INTEGER PRIMARY KEY, started_at REAL, deadline_at REAL, ended_at REAL,
        status TEXT, pause_reason TEXT, initial_completed INTEGER, final_completed INTEGER, workers INTEGER)''')
    fields=['wave_id','started_at','deadline_at','ended_at','status','pause_reason','initial_completed','final_completed','workers']
    row=connection.execute('SELECT * FROM waves ORDER BY wave_id DESC LIMIT 1').fetchone()
    last=dict(zip(fields,row)) if row else None
    if last and not start_next:
        resumable=last['status'] in {'RUNNING','INTERRUPTED'} or (
            last['status']=='PAUSED' and last['pause_reason'] in {'LOW_MEMORY','LOW_DISK'})
        if not resumable or now>=last['deadline_at']:
            raise RuntimeError('Previous wave ended. Wait for user authorization, then use --start-next-wave.')
        connection.execute("UPDATE waves SET status='RUNNING',pause_reason=NULL,workers=? WHERE wave_id=?",(workers,last['wave_id']))
        connection.commit();last.update(status='RUNNING',pause_reason=None,workers=workers)
        return last
    if last and last['status'] in {'RUNNING','INTERRUPTED'}:
        connection.execute("UPDATE waves SET status='INTERRUPTED',ended_at=? WHERE wave_id=?",(now,last['wave_id']))
    number=1 if last is None else last['wave_id']+1
    wave=dict(wave_id=number,started_at=now,deadline_at=now+hours*3600,ended_at=None,
              status='RUNNING',pause_reason=None,initial_completed=completed,final_completed=None,workers=workers)
    connection.execute('INSERT INTO waves VALUES(?,?,?,?,?,?,?,?,?)',tuple(wave[k] for k in fields))
    connection.commit();return wave


def end_wave(connection,wave,status,reason,completed,*,now=None):
    now=time.time() if now is None else now
    connection.execute('UPDATE waves SET status=?,pause_reason=?,ended_at=?,final_completed=? WHERE wave_id=?',
                       (status,reason,now,completed,wave['wave_id']))
    connection.commit()
