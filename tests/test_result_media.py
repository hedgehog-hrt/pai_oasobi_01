import asyncio,json
from pathlib import Path
import numpy as np
import pytest
from rosa_repro.explanation_video import record,inspect,export
from rosa_repro.dummy import observation
from rosa_repro.paired import digest_value

class Sim:
 async def reset(self,state,seed):return dict(observation=observation(1),restored_state_sha256=digest_value(state))
 async def step(self,a,n):return dict(observation=observation(2),steps=n,done=True,success=True)
async def predict(obs):return {'action.x':np.ones((1,8,1),np.float32)}


def test_separate_recording_and_gif_roundtrip(tmp_path):
 c=asyncio.run(record(Sim(),predict,tmp_path/'capture',state=np.array([1]),seed=42,mode='cpu_dummy',provenance={}))
 assert c['performance_eligible'] is False and c['episode_done'] and len(c['frames'])==2
 r=export(tmp_path/'capture',tmp_path/'video',camera='video.image');assert r['frames']==2
 from PIL import Image
 with Image.open(r['video']) as im:assert im.n_frames==2
 with pytest.raises(ValueError):export(tmp_path/'capture',tmp_path/'wrong',camera='missing')
 (tmp_path/'capture/frame-00000/arrays.npz').write_bytes(b'changed')
 with pytest.raises(ValueError,match='frame changed'):inspect(tmp_path/'capture')


def test_capture_budget_and_gpu_identity(tmp_path):
 with pytest.raises(ValueError):asyncio.run(record(Sim(),predict,tmp_path/'gpu',state=np.array([1]),seed=42,mode='gpu',provenance={}))
 with pytest.raises(ValueError,match='byte budget'):asyncio.run(record(Sim(),predict,tmp_path/'small',state=np.array([1]),seed=42,mode='cpu_dummy',provenance={},max_bytes=1024))
 assert not (tmp_path/'small/capture.json').exists()


def test_missing_figure_cells_not_zero(tmp_path):
 from rosa_repro.result_figures import data
 expected=dict(execution_mode='cpu_dummy',action_semantics='virtual_timed_chunks',source_sha256='a'*64,workload_sha256='b'*64,protocol_sha256='c'*64)
 m=dict(schema_version='1.0',purpose='prospective_pipeline_comparison',conditions={'c':expected},cells=[dict(id='one',condition_id='c',variant='ROSA',repetition=0,directory='absent')])
 r,series,label=data(dict(kind='pipeline_comparison',manifest=m),tmp_path)
 assert r['status']=='incomplete' and series[0]['points'][0]['value'] is None and 'CPU' in label
