"""Immutable experiment matrices and single-writer, crash-resumable results."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import re
from pathlib import Path
import sqlite3

from .regression import sha256,write_json


def stable_id(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def protocol(phase):
    if phase=='formal':
        return dict(warmup_min_s=30.,warmup_min_success=20,warmup_max_s=180.,
                    measure_min_s=120.,measure_min_success=1000,measure_max_s=900.,request_timeout_s=10.)
    if phase=='cpu_smoke':
        return dict(warmup_min_s=.1,warmup_min_success=4,warmup_max_s=5.,
                    measure_min_s=.3,measure_min_success=8,measure_max_s=3.,request_timeout_s=5.)
    if phase=='exploratory':
        return dict(warmup_min_s=30.,warmup_min_success=20,warmup_max_s=180.,
                    measure_min_s=30.,measure_min_success=1,measure_max_s=30.,request_timeout_s=10.)
    raise ValueError('Unknown measurement phase')


def build_matrix(*,capacities,observation_fingerprint,runtime_fingerprint,
                 execution_kind='cpu_dummy',phase='cpu_smoke',batches=(1,2,4,8,16),
                 repetitions=3,robots=(1,2,4,8,16,32,64),rate_cap_hz=None):
    """C_b is explicit input evidence, never an inferred paper performance value.

    Missing capacities produce unresolved R03 cells. R04 no-cap and explicit
    cap are generated; a cap is not called ROSA-optimal without a scheduler run.
    """
    if execution_kind not in ('cpu_dummy','gpu_replay'):raise ValueError('Invalid execution kind')
    if phase=='cpu_smoke' and execution_kind!='cpu_dummy':raise ValueError('Smoke protocol is CPU only')
    if type(repetitions) is not int or repetitions<1:raise ValueError('Invalid repetition count')
    if any(type(b) is not int or not 1<=b<=1024 for b in batches):raise ValueError('Invalid batch')
    if any(type(k) is not int or not 1<=k<=1024 for k in robots):raise ValueError('Invalid robot count')
    if not observation_fingerprint or not runtime_fingerprint:raise ValueError('Missing fingerprints')
    if rate_cap_hz is not None and (not math.isfinite(rate_cap_hz) or rate_cap_hz<=0):raise ValueError('Invalid rate cap')
    cases=[];unresolved=[]
    for b in batches:
        cap=capacities.get(str(b))
        if cap is None:
            unresolved.append(dict(stage='R03',batch_size=b,reason='capacity_not_measured'))
        else:
            rate=cap['hz']
            if (not math.isfinite(rate) or rate<=0 or not re.fullmatch('[0-9a-f]{64}',cap.get('evidence_sha256',''))
                    or cap.get('execution_kind')!=execution_kind):
                raise ValueError('Capacity value/evidence/execution-kind mismatch')
            for load in range(1,11):
                for rep in range(repetitions):
                    cases.append(dict(stage='R03',mode='open_loop',batch_size=b,batch_wait_s=.005,
                        offered_hz=rate*load/10,load_fraction=load/10,capacity_evidence=cap,
                        robots=64,action_time_s=0.,rate_cap_hz=None,seed=42+rep,rep=rep))
        for k in robots:
            for cap_hz in ([None,rate_cap_hz] if rate_cap_hz is not None else [None]):
                for rep in range(repetitions):
                    cases.append(dict(stage='R04',mode='closed_loop',batch_size=b,batch_wait_s=.005,
                        offered_hz=None,robots=k,action_time_s=.2,rate_cap_hz=cap_hz,
                        rate_selection='explicit_input' if cap_hz else 'uncapped',seed=42+rep,rep=rep))
    for case in cases:
        case.update(execution_kind=execution_kind,phase=phase,protocol=protocol(phase),deadline_ms=200.,
                    observation_fingerprint=observation_fingerprint,runtime_fingerprint=runtime_fingerprint)
        case['case_id']=case['stage'].lower()+'-'+stable_id(case)[:16]
    if len({c['case_id'] for c in cases})!=len(cases):raise ValueError('Duplicate experiment conditions')
    return dict(schema_version='1.0',cases=cases,unresolved=unresolved,
                timing_budget_s=dict(measure_min=sum(c['protocol']['measure_min_s'] for c in cases),
                    upper_excluding_load_compile_drain=sum(c['protocol']['warmup_max_s']+c['protocol']['measure_max_s'] for c in cases)),
                cost=None,cost_note='No current GPU price supplied; no cost or GPU execution authorized by this manifest')


class ExperimentStore:
    """Single driver flock; interrupted attempts retained, never overwritten."""
    def __init__(self,root,matrix):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'.runner.lock').open('a')
        try:fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close();raise RuntimeError('Another driver owns this experiment')
        self.db=None
        try:
            manifest=self.root/'matrix.json'
            if manifest.exists():
                if stable_id(json.loads(manifest.read_text()))!=stable_id(matrix):
                    raise ValueError('Resume matrix/fingerprint mismatch')
            else:write_json(manifest,matrix)
            self.db=sqlite3.connect(self.root/'state.sqlite3')
            self.db.execute('CREATE TABLE IF NOT EXISTS attempts (case_id TEXT, number INTEGER, status TEXT, result_sha TEXT, PRIMARY KEY(case_id,number))')
            self.db.execute("UPDATE attempts SET status='interrupted' WHERE status='running'")
            self.db.commit()
        except BaseException:
            self.close();raise

    def completed(self,case_id):
        row=self.db.execute("SELECT number,result_sha FROM attempts WHERE case_id=? AND status='complete' ORDER BY number DESC LIMIT 1",(case_id,)).fetchone()
        if row is None:return False
        path=self.root/case_id/f'attempt-{row[0]:03d}'/'artifact_hashes.json'
        if not path.exists() or sha256(path)!=row[1]:raise ValueError('Completed artifact manifest missing/modified')
        for name,digest in json.loads(path.read_text()).items():
            file=path.parent/name
            if not file.is_file() or sha256(file)!=digest:raise ValueError('Completed artifact missing/modified')
        return True

    @contextmanager
    def attempt(self,case_id):
        if not case_id or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in case_id):
            raise ValueError('Invalid case identifier')
        number=self.db.execute('SELECT COALESCE(MAX(number),0)+1 FROM attempts WHERE case_id=?',(case_id,)).fetchone()[0]
        directory=self.root/case_id/f'attempt-{number:03d}'
        # Reserve the attempt before creating artifacts; a process death in
        # between is recovered as interrupted, and the next number is unused.
        self.db.execute('INSERT INTO attempts VALUES (?,?,?,NULL)',(case_id,number,'running'));self.db.commit()
        try:
            directory.mkdir(parents=True,exist_ok=False)
            yield directory
            if not (directory/'result.json').is_file():raise ValueError('Trial did not finalize result.json')
            hashes={str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}
            write_json(directory/'artifact_hashes.json',hashes)
            self.db.execute('UPDATE attempts SET status=?,result_sha=? WHERE case_id=? AND number=?',
                            ('complete',sha256(directory/'artifact_hashes.json'),case_id,number));self.db.commit()
        except BaseException:
            self.db.execute('UPDATE attempts SET status=? WHERE case_id=? AND number=?',('failed',case_id,number));self.db.commit()
            raise

    def report(self):
        rows=self.db.execute('SELECT case_id,number,status FROM attempts ORDER BY case_id,number').fetchall()
        return [dict(case_id=c,attempt=n,status=s) for c,n,s in rows]

    def close(self):
        if self.db is not None:self.db.close()
        fcntl.flock(self.lock,fcntl.LOCK_UN);self.lock.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def summarize_store(root):
    """Derived, overwriteable summary; every raw artifact is hash-checked first."""
    import csv
    root=Path(root)
    matrix=json.loads((root/'matrix.json').read_text())
    rows=[]
    with ExperimentStore(root,matrix) as store:
        for case in matrix['cases']:
            row={k:case.get(k) for k in ('case_id','stage','mode','execution_kind','batch_size','robots','offered_hz','rate_cap_hz','rep')}
            row.update(status='not_run',requests=None,qualified_action_rate_hz=None,p99_ms=None)
            if store.completed(case['case_id']):
                number=store.db.execute("SELECT MAX(number) FROM attempts WHERE case_id=? AND status='complete'",(case['case_id'],)).fetchone()[0]
                result=json.loads((root/case['case_id']/f'attempt-{number:03d}'/'result.json').read_text())
                metrics=result.get('metrics',{})
                row.update(status=result['status'],requests=metrics.get('requests'),
                    qualified_action_rate_hz=metrics.get('qualified_action_rate_hz'),
                    p99_ms=metrics.get('latency_ms_successful_valid_only',{}).get('p99'))
            rows.append(row)
        report=dict(cases=rows,unresolved=matrix.get('unresolved',[]),attempts=store.report(),
                    note='CPU dummy and measurement-shortfall rows must not be treated as GPU performance evidence')
    (root/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if rows:
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return report


def split_observation_manifest(root,seed=42):
    """Episode/state-disjoint profile and evaluation sets, grouping both arms."""
    import random
    root=Path(root);groups={}
    for path in sorted(root.rglob('observation.json')):
        manifest=json.loads(path.read_text());provenance=manifest['provenance']
        state=provenance['case']['state_sha256']
        key=stable_id({'task':provenance['task'],'state_sha256':state})
        if sha256(path.parent/'arrays.npz')!=manifest['arrays_sha256']:raise ValueError('Observation data hash mismatch')
        groups.setdefault(key,[]).append(dict(path=str(path.relative_to(root)),manifest_sha256=sha256(path)))
    if len(groups)<2:raise ValueError('At least two distinct initial states required for a held-out split')
    keys=sorted(groups);random.Random(seed).shuffle(keys);cut=len(keys)//2
    return dict(schema_version='1.0',seed=seed,grouping='task + initial-state hash, arms kept together',
        profile={k:groups[k] for k in keys[:cut]},evaluation={k:groups[k] for k in keys[cut:]})
