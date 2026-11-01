import io,threading
from rosa_repro.gpu_telemetry_stream import GPUStream

class Process:
    def __init__(self,line):self.stdout=io.StringIO(line);self.returncode=None;self.terminated=False
    def poll(self):return self.returncode
    def terminate(self):self.terminated=True;self.returncode=0
    def wait(self,timeout=None):return self.returncode

def test_one_stream_multiple_reads_stale_dead_and_cleanup():
    process=Process('GPU-abcd, GPU, 1.0, 1000, 100, 10\n');clock=[10**9];calls=[]
    def spawn(*a,**k):calls.append(a);return process
    s=GPUStream({'uuid':'GPU-abcd'},popen=spawn,clock=lambda:clock[0]);s.ready.wait(1)
    assert s.sample()['kind']=='nvidia_smi_sample' and s.sample()['used_mib']==100 and len(calls)==1
    clock[0]+=3001000000;assert s.sample()['error_type']=='GPUStreamStale'
    clock[0]=10**9;process.returncode=2;assert s.sample()['kind']=='unavailable'
    process.returncode=None;s.close();assert process.terminated and process.stdout.closed and s.sample()['kind']=='unavailable'

def test_wrong_identity_and_malformed_values_fail_closed():
    for line in ['GPU-other, GPU, 1, 1000, 100, 10\n','GPU-abcd, GPU, 1, 1000, 100, nan\n']:
        p=Process(line);s=GPUStream({'uuid':'GPU-abcd'},popen=lambda *a,**k:p);s.ready.wait(1)
        assert s.sample()['kind']=='unavailable';s.close()
