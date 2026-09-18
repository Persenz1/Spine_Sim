import sqlite3
import pytest
from spine_sim.wave_control import begin_wave,end_wave


def test_wave_closes_and_next_wave_requires_explicit_start():
    db=sqlite3.connect(':memory:')
    first=begin_wave(db,8,16,68,now=1000)
    assert first['deadline_at']==29800
    end_wave(db,first,'PAUSED','TIME_LIMIT',8000,now=29900)
    with pytest.raises(RuntimeError,match='authorization'):
        begin_wave(db,8,16,8000,now=30000)
    second=begin_wave(db,8,16,8000,start_next=True,now=30000)
    assert second['wave_id']==2
    assert second['deadline_at']==58800


def test_restart_within_wave_does_not_extend_deadline():
    db=sqlite3.connect(':memory:')
    first=begin_wave(db,8,16,0,now=1000)
    end_wave(db,first,'INTERRUPTED','PROCESS_INTERRUPTED',20,now=1100)
    resumed=begin_wave(db,8,12,20,now=1200)
    assert resumed['wave_id']==1
    assert resumed['deadline_at']==first['deadline_at']
    assert resumed['initial_completed']==0
    with pytest.raises(RuntimeError):begin_wave(db,8,12,20,now=30000)


def test_user_pause_is_not_an_automatic_resume():
    db=sqlite3.connect(':memory:')
    wave=begin_wave(db,8,16,0,now=1000)
    end_wave(db,wave,'PAUSED','USER_REQUEST',20,now=1100)
    with pytest.raises(RuntimeError):begin_wave(db,8,16,20,now=1200)
