"""Windows physical-core and RAM budgets; BLAS remains one thread per worker."""
import ctypes,os


def physical_cores():
    # RelationProcessorCore entries are variable-length; count headers only.
    kernel=ctypes.windll.kernel32;size=ctypes.c_ulong()
    kernel.GetLogicalProcessorInformationEx(0,None,ctypes.byref(size))
    buffer=ctypes.create_string_buffer(size.value)
    if not kernel.GetLogicalProcessorInformationEx(0,buffer,ctypes.byref(size)):
        raise ctypes.WinError()
    offset=0;count=0
    while offset<size.value:
        relation=ctypes.c_ulong.from_buffer(buffer,offset).value
        length=ctypes.c_ulong.from_buffer(buffer,offset+4).value
        if relation==0:count+=1
        offset+=length
    return count


def memory_status():
    class Status(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in
                   ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    status=Status();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise ctypes.WinError()
    return status.total,status.available


def hardware():
    total,available=memory_status();cores=physical_cores();gib=2**30
    # On 128 GiB, leave at least 38 GiB to Windows / other tasks. On smaller
    # hosts leave 8 GiB. Small-domain generation/build gets another 4 GiB;
    # each small-array worker is budgeted at 512 MiB, separate from shared fields.
    budget=min(90*gib,max(0,total-8*gib));reserve=total-budget
    workers=max(1,min(os.cpu_count() or 1,int((available-reserve-4*gib)//(gib//2))))
    return dict(cpus=os.cpu_count() or 1,physical_cores=cores,total_memory=total,
                available_memory=available,reserve_bytes=reserve,memory_budget_bytes=budget,auto_workers=workers,
                initial_workers=min(workers,cores))


class CpuUsage:
    def __init__(self):self.previous=None

    def sample(self):
        idle=ctypes.c_ulonglong();kernel=ctypes.c_ulonglong();user=ctypes.c_ulonglong()
        if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle),ctypes.byref(kernel),ctypes.byref(user)):
            raise ctypes.WinError()
        current=(idle.value,kernel.value+user.value);old=self.previous;self.previous=current
        if old is None or current[1]==old[1]:return None
        return 100*(1-(current[0]-old[0])/(current[1]-old[1]))


def ramp_workers(active,maximum,cpu_percent,available,reserve):
    """Increase only while system CPU has spare capacity and RAM covers new jobs."""
    if cpu_percent is None or cpu_percent>=95:return active
    room=max(0,int((available-reserve-8*2**30)//2**30))
    return min(maximum,active+min(max(1,maximum//8),room))
