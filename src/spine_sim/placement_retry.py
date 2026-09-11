"""Bounded, ordered placement retries after failed preload only."""
import time
import numpy as np


def run_with_placements(array,start_xy,preload,distance,step,*,offsets_m,
                        attempt_cpu_budget_s,total_cpu_budget_s,preload_min_divisor=128,progress=None):
    started=time.process_time();attempts=[];selected=None;search_status='EXHAUSTED'
    last=dict(status='BALANCED_PLACEMENT_DOMAIN_LIMIT',phase='preload',rows=[])
    nominal=array.guides+array.length[:,None]*array.axes
    margin=float(np.max(array.radius))
    for index,offset in enumerate(offsets_m):
        remaining=total_cpu_budget_s-(time.process_time()-started)
        if remaining<=0:
            search_status='CPU_BUDGET_LIMIT';break
        start=np.asarray(start_xy,float)+np.asarray(offset,float)
        xy=nominal[:,:2]+start
        inside=(xy[:,0].min()-margin>=array.surface.x[0] and
                xy[:,0].max()+distance+margin<=array.surface.x[-1] and
                xy[:,1].min()-margin>=array.surface.y[0] and xy[:,1].max()+margin<=array.surface.y[-1])
        entry=dict(index=index,offset_xy_m=list(offset),start_xy_m=start.tolist())
        if not inside:
            attempts.append(dict(entry,status='PLACEMENT_OUTSIDE_DOMAIN',phase='preload',cpu_s=0.))
            continue
        def report(row):
            if progress:progress(dict(row,placement_index=index,start_xy_m=start.tolist()))
        tick=time.process_time()
        # run() always creates a fresh first-contact state: no forces, friction
        # history, or rows from the previous placement are carried across.
        last=array.run(start,preload,distance,step,progress=report,
                       cpu_budget_s=min(attempt_cpu_budget_s,remaining),preload_min_divisor=preload_min_divisor)
        failed_preload=(last.get('phase')=='preload' and last['status'] in
                        {'BALANCED_NUMERICAL_FAILURE','BALANCED_AXIAL_RANGE','BALANCED_BUDGET_LIMIT'})
        entry.update(status=last['status'],phase=last.get('phase','drag'),
                     reason=last.get('reason'),cpu_s=time.process_time()-tick,
                     preload_failed=failed_preload)
        if failed_preload:
            entry['failed_result']=last
        attempts.append(entry)
        if not failed_preload:
            selected=index;search_status='ORIGINAL' if index==0 else 'RELOCATED'
            break
    return dict(last,placement_attempts=attempts,selected_placement_index=selected,
                selected_start_xy_m=attempts[-1]['start_xy_m'] if selected is not None else None,
                placement_search_status=search_status,total_cpu_s=time.process_time()-started)
