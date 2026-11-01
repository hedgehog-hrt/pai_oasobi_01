from rosa_repro.multicomponent import paper_class

NS=1_000_000_000

def req(q,component,sent,finished=None,scheduled=None,slot=None,plan=None,status='ok',valid=True,warmup=False):
    deadline={'s1':200,'s2':2000,'safety':500,'monitor':2000}[component]
    return dict(schema_version='2.0',record_type='request',run_id='run',clock_domain='clock',robot_id='r0',episode_id='e0',
        request_id=q,attempt_id=q+'-a',component_id=component,observation_seq=0,
        scheduled_ns=round((sent if scheduled is None else scheduled)*NS),sent_ns=round(sent*NS),
        finished_ns=round((sent+.05 if finished is None else finished)*NS),deadline_ms=deadline,
        status=status,valid_output=valid,warmup=warmup,batch_id=None,actual_batch_size=None,slot_index=slot,plan_request_id=plan)

def action(r,*,stopped=False,start=None):
    t=r['finished_ns'] if start is None else round(start*NS)
    return dict(schema_version='2.0',record_type='action',**{k:r[k] for k in
        ('run_id','clock_domain','robot_id','episode_id','request_id','component_id')},
        action_id='act-'+r['request_id'],start_ns=t,end_ns=t+10_000_000,
        applied=not stopped,executed_steps=0 if stopped else 8,safety_stopped=stopped)

def fixture(pipeline='P4',count=1):
    cls=paper_class('c0',pipeline)
    records=[]
    if pipeline!='P1':records.append(req('monitor-0','monitor',0,slot=0))
    if pipeline in ('P3','P4'):
        records += [req(f'safety-{i}','safety',i*.5,slot=i) for i in range(8)]
    if pipeline=='P4':records.append(req('plan','s2',0))
    for i in range(count):
        r=req(f's1-{i}','s1',2+i*.1,plan='plan' if pipeline=='P4' else None)
        records += [r,action(r)]
    return dict(schema_version='2.0',run_id='run',clock_domain='clock',run_epoch_ns=0,
        window_start_ns=2*NS,window_end_ns=4*NS,audit_end_ns=10*NS,classes=[cls],
        episodes=[dict(class_id='c0',robot_id='r0',episode_id='e0',start_ns=0,end_ns=10*NS)],records=records)
