"""One UUID-bound nvidia-smi stream per physical actor; stale data is failure."""
import csv,math,re,subprocess,threading,time


class GPUStream:
    def __init__(self,properties,*,popen=subprocess.Popen,clock=time.monotonic_ns):
        self.properties=properties;self.clock=clock;self.latest=None;self.error=None
        self.process=None;self.thread=None;self.ready=threading.Event();self.closed=False
        if properties is None:return
        self.uuid=properties['uuid']
        self.query_uuid='GPU-'+self.uuid if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',self.uuid) else self.uuid
        if not self.query_uuid.startswith('GPU-'):raise ValueError('Physical GPU UUID unavailable')
        self.process=popen(['nvidia-smi','--id='+self.query_uuid,
            '--query-gpu=uuid,name,driver_version,memory.total,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits','--loop-ms=1000'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
        self.thread=threading.Thread(target=self._read,daemon=True);self.thread.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                row=next(csv.reader([line]));row=[s.strip() for s in row]
                if len(row)!=6 or row[0]!=self.query_uuid:raise ValueError('GPU stream identity mismatch')
                n=[float(row[i]) for i in (3,4,5)]
                if not all(math.isfinite(x) and x>=0 for x in n) or n[2]>100 or n[1]>n[0]:raise ValueError('Invalid GPU stream values')
                self.latest=(self.clock(),dict(kind='nvidia_smi_sample',uuid=self.uuid,nvidia_smi_uuid=row[0],name=row[1],driver=row[2],
                    total_mib=n[0],used_mib=n[1],utilization_pct=n[2],scope='whole device sample, including other processes; not allocation peak'))
                self.ready.set()
        except Exception as exc:self.error=type(exc).__name__
        finally:self.ready.set()

    def sample(self):
        if self.properties is None:return dict(kind='cpu_no_gpu',gpu_execution=False)
        self.ready.wait(3)
        if self.closed or self.error or self.process.poll() is not None or self.latest is None:
            return dict(kind='unavailable',error_type=self.error or 'GPUStreamUnavailable')
        stamp,value=self.latest;age=(self.clock()-stamp)/1e6
        if not 0<=age<=3000:return dict(kind='unavailable',error_type='GPUStreamStale',sample_age_ms=age)
        return dict(value,collection='persistent_nvidia_smi_1000ms',captured_monotonic_ns=stamp,sample_age_ms=age)

    def close(self):
        self.closed=True
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
            if self.thread:self.thread.join(timeout=3)
            self.process.stdout.close()
