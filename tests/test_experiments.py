import json
from pathlib import Path
import sqlite3
import pytest
from rosa_repro.experiments import build_matrix,protocol,ExperimentStore,stable_id
from rosa_repro.regression import write_json


def matrix():
    return build_matrix(capacities={'1':dict(hz=100.,evidence_sha256='a'*64,execution_kind='cpu_dummy')},
        observation_fingerprint='obs',runtime_fingerprint='runtime',batches=(1,2),robots=(1,4),repetitions=2,rate_cap_hz=2.)


def test_matrix_exact_load_grid_and_unmeasured_capacities():
    m=matrix();p=[c for c in m['cases'] if c['stage']=='R03']
    assert len(p)==20 and sorted({c['load_fraction'] for c in p})==[i/10 for i in range(1,11)]
    assert len([c for c in m['cases'] if c['stage']=='R04'])==16
    assert m['unresolved']==[dict(stage='R03',batch_size=2,reason='capacity_not_measured')]
    assert len({c['case_id'] for c in m['cases']})==36


def test_formal_protocol_matches_frozen_plan():
    p=protocol('formal')
    assert (p['warmup_min_s'],p['warmup_min_success'],p['warmup_max_s'])==(30,20,180)
    assert (p['measure_min_s'],p['measure_min_success'],p['measure_max_s'])==(120,1000,900)


def test_cpu_capacity_cannot_be_promoted_to_gpu_evidence():
    with pytest.raises(ValueError,match='execution-kind'):
        build_matrix(capacities={'1':dict(hz=100,evidence_sha256='a'*64,execution_kind='cpu_dummy')},
            observation_fingerprint='obs',runtime_fingerprint='rt',execution_kind='gpu_replay',phase='formal')


def test_resume_skips_verified_complete_and_preserves_failed_attempt(tmp_path):
    m=matrix();cid=m['cases'][0]['case_id']
    with ExperimentStore(tmp_path,m) as s:
        with pytest.raises(RuntimeError):
            with s.attempt(cid) as out:
                (out/'events.jsonl').write_text('partial\n');raise RuntimeError('interrupted test')
        assert not s.completed(cid)
    with ExperimentStore(tmp_path,m) as s:
        with s.attempt(cid) as out:write_json(out/'result.json',{'status':'complete'})
        assert out.name=='attempt-002' and s.completed(cid)
        assert [r['status'] for r in s.report()]==['failed','complete']
    assert (tmp_path/cid/'attempt-001/events.jsonl').read_text()=='partial\n'


def test_crashed_running_attempt_is_marked_interrupted_on_reopen(tmp_path):
    m=matrix();cid=m['cases'][0]['case_id']
    with ExperimentStore(tmp_path,m) as s:
        s.db.execute('INSERT INTO attempts VALUES (?,?,?,NULL)',(cid,1,'running'));s.db.commit()
    with ExperimentStore(tmp_path,m) as s:assert s.report()[0]['status']=='interrupted'


def test_results_tampering_and_changed_matrix_cannot_silently_resume(tmp_path):
    m=matrix();cid=m['cases'][0]['case_id']
    with ExperimentStore(tmp_path,m) as s:
        with s.attempt(cid) as out:write_json(out/'result.json',{'status':'complete'})
        (out/'result.json').write_text('{}')
        with pytest.raises(ValueError,match='modified'):s.completed(cid)
    with pytest.raises(ValueError,match='fingerprint'):ExperimentStore(tmp_path,dict(m,changed=True))


def test_double_driver_and_unfinalized_trial_rejected(tmp_path):
    m=matrix();cid=m['cases'][0]['case_id']
    with ExperimentStore(tmp_path,m) as s:
        with pytest.raises(RuntimeError,match='driver'):ExperimentStore(tmp_path,m)
        with pytest.raises(ValueError,match='finalize'):
            with s.attempt(cid):pass


def test_summary_keeps_missing_and_insufficient_rows(tmp_path):
    from rosa_repro.experiments import summarize_store
    m=matrix();cid=m['cases'][0]['case_id']
    with ExperimentStore(tmp_path,m) as s:
        with s.attempt(cid) as out:write_json(out/'result.json',{'status':'insufficient_or_failed'})
    r=summarize_store(tmp_path)
    assert r['cases'][0]['status']=='insufficient_or_failed'
    assert all(row['requests'] is None for row in r['cases'])
    assert r['cases'][1]['status']=='not_run' and (tmp_path/'summary.csv').exists()


def test_observation_split_groups_both_arms_by_state(tmp_path):
    import numpy as np
    from rosa_repro.regression import save_observation
    from rosa_repro.dummy import observation
    from rosa_repro.experiments import split_observation_manifest
    for i in range(4):
        for arm in ('official','shared'):
            save_observation(tmp_path/str(i)/arm,observation(i),
                {'task':'T01','case':{'state_sha256':str(i)}})
    result=split_observation_manifest(tmp_path)
    assert len(result['profile'])==len(result['evaluation'])==2
    assert set(result['profile']).isdisjoint(result['evaluation'])
    assert all(len(v)==2 for v in result['profile'].values())
