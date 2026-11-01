"""Separate explanation-run capture and offline GIF export; never a performance trial."""
import asyncio,json,math,time,uuid
from pathlib import Path
import numpy as np
from .regression import save_observation,load_observation,write_json,sha256
from .execution_plan import runtime_source_manifest,digest


async def record(simulator,predict,output,*,state,seed,mode,provenance,max_frames=120,max_bytes=128*1024*1024,requested_steps=8,timeout_s=60.):
    if mode not in ('cpu_dummy','gpu') or type(max_frames) is not int or not 2<=max_frames<=1000 or not 1024<=max_bytes<=512*1024*1024:raise ValueError('Invalid bounded capture')
    if not math.isfinite(timeout_s) or not 0<timeout_s<=600 or type(requested_steps) is not int or not 1<=requested_steps<=100:raise ValueError('Invalid recording limits')
    if mode=='gpu' and (provenance.get('model_execution')!='external_model_cuda' or not provenance.get('model_revision') or not provenance.get('simulator_revision')):raise ValueError('Explicit model/simulator provenance required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);frames=[];size=0;step=0;done=False;success=False
    begin=time.monotonic_ns();reset=await asyncio.wait_for(simulator.reset(state,seed),timeout_s);obs=reset['observation']
    if reset['restored_state_sha256']!=__import__('rosa_repro.paired',fromlist=['digest_value']).digest_value(state):raise ValueError('Recording initial state differs')
    async def bounded(value):return await asyncio.wait_for(value,timeout_s)
    for index in range(max_frames):
        name=f'frame-{index:05d}';manifest=save_observation(out/name,obs,dict(purpose='explanation_only',run_id=out.name,step=step,mode=mode))
        size+=sum(p.stat().st_size for p in (out/name).iterdir())
        if size>max_bytes:raise ValueError('Recording byte budget exceeded; partial capture is not exportable')
        frames.append(dict(directory=name,step=step,captured_ns=time.monotonic_ns(),observation_sha256=sha256(out/name/'observation.json'),arrays_sha256=manifest['arrays_sha256']))
        if done:break
        if index==max_frames-1:break
        action=await bounded(predict(obs));result=await bounded(simulator.step(action,requested_steps))
        step+=result['steps'];obs=result['observation'];done=bool(result['done']);success=bool(result['success'])
    value=dict(kind='explanation_capture',mode=mode,purpose='explanation_only',performance_eligible=False,source_manifest=runtime_source_manifest(),
        run_id=uuid.uuid4().hex,provenance=provenance,seed=seed,initial_state_sha256=reset['restored_state_sha256'],frames=frames,
        elapsed_s=(time.monotonic_ns()-begin)/1e9,episode_done=done,success_reported_by_simulator=success,frame_limit_reached=not done,max_frames=max_frames,max_bytes=max_bytes,bytes_saved=size)
    write_json(out/'capture.json',value);return value


def inspect(directory):
    p=Path(directory);c=json.loads((p/'capture.json').read_text())
    if c['kind']!='explanation_capture' or c['purpose']!='explanation_only' or c['performance_eligible'] is not False or c['mode'] not in ('gpu','cpu_dummy'):raise ValueError('Explanation-only capture required')
    if not 1<=len(c['frames'])<=1000 or len({f['directory'] for f in c['frames']})!=len(c['frames']):raise ValueError('Invalid capture frame inventory')
    last=-1;lasttime=-1
    for f in c['frames']:
        d=(p/f['directory']).resolve()
        if not d.is_relative_to(p.resolve()) or d.is_symlink() or sha256(d/'observation.json')!=f['observation_sha256'] or sha256(d/'arrays.npz')!=f['arrays_sha256']:raise ValueError('Capture frame changed')
        if f['step']<last or f['captured_ns']<lasttime:raise ValueError('Capture order changed')
        load_observation(d);last=f['step'];lasttime=f['captured_ns']
    return c


def export(directory,output,*,camera,fps=5.):
    from PIL import Image,ImageDraw
    if not math.isfinite(fps) or not .1<=fps<=30:raise ValueError('Playback fps must be 0.1..30')
    c=inspect(directory);p=Path(directory);out=Path(output);out.mkdir(parents=True,exist_ok=False);frames=[];decoded_bytes=0
    for i,f in enumerate(c['frames']):
        obs,_=load_observation(p/f['directory']);a=obs.get(camera)
        if not isinstance(a,np.ndarray) or a.ndim!=5 or a.shape[0]!=1 or a.shape[-1]!=3 or a.dtype!=np.uint8 or a[0,-1].size>6*1024*1024:raise ValueError('Require bounded batch-one uint8 RGB video camera')
        decoded_bytes+=a[0,-1].nbytes
        if decoded_bytes>256*1024*1024:raise ValueError('Decoded video memory budget exceeded')
        im=Image.fromarray(a[0,-1]);im.thumbnail((960,720));canvas=Image.new('RGB',(max(520,im.width),im.height+90),'#101c29');canvas.paste(im,((canvas.width-im.width)//2,70));draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f"{c['mode'].upper()} | EXPLANATION ONLY | frame {i}, simulator step {f['step']}",fill='white')
        draw.text((12,28),'Fixed-rate playback. Not real-time speed or performance evidence.',fill='#fbbf78')
        frames.append(canvas)
    frames[0].save(out/'explanation.gif',save_all=True,append_images=frames[1:],duration=round(1000/fps),loop=0,optimize=False,disposal=2)
    with Image.open(out/'explanation.gif') as check:
        if check.n_frames!=len(frames):raise ValueError('Encoded frame count differs')
    write_json(out/'video.json',dict(kind='explanation_video',capture_sha256=sha256(p/'capture.json'),gif_sha256=sha256(out/'explanation.gif'),mode=c['mode'],
        performance_eligible=False,camera=camera,fps=fps,frames=len(frames),playback='fixed frame rate, not wall-clock or simulator-frequency reproduction'))
    return dict(status='complete',video=str(out/'explanation.gif'),frames=len(frames))
