"""Bounded CPU fault-injection scenario for the real control and transport code."""
import asyncio,math
from .dynamic_control import validate
from .dummy import observation


def validate_scenario(spec,steps):
    validate(spec)
    if spec['blueprint']['mode']!='cpu_dummy':raise ValueError('Scenario harness only accepts CPU dummy; GPU actions require a real simulator adapter')
    if not isinstance(steps,list) or not 1<=len(steps)<=256:raise ValueError('Require 1..256 predeclared steps')
    robots={r['robot_id'] for c in spec['blueprint']['classes'] for r in c['robots']};workers={w['worker_id'] for w in spec['workers']};aliases=set()
    for step in steps:
        op=step.get('op')
        if op not in ('predict','apply','join','leave','fault','failover','health','wait'):raise ValueError('Unknown scenario operation')
        if op in ('predict','join','leave') and step.get('robot') not in robots:raise ValueError('Unknown robot')
        if op in ('fault','failover') and step.get('worker') not in workers:raise ValueError('Unknown worker')
        if op=='predict':
            if step.get('as') in aliases or not isinstance(step.get('as'),str):raise ValueError('Unique prediction alias required')
            c=next(c for c in spec['blueprint']['classes'] if any(r['robot_id']==step['robot'] for r in c['robots']))
            if step.get('component') not in {m['id'] for m in c['audit']['components']}:raise ValueError('Unknown component')
            aliases.add(step['as'])
        if op=='apply' and step.get('ticket') not in aliases:raise ValueError('Prediction must precede apply')
        if op=='wait' and (type(step.get('seconds')) not in (int,float) or not math.isfinite(step['seconds']) or not 0<=step['seconds']<=10):raise ValueError('Wait must be 0..10 seconds')
        if op not in ('wait','health') and not isinstance(step.get('expect'),str):raise ValueError('Declare expected outcome')
    return steps


async def run(controller,transport,steps):
    validate_scenario(controller.spec,steps);tickets={};results=[]
    async def execute_action(rid,output,duration):
        await asyncio.sleep(duration);return min(x.shape[1] for x in output.values())
    for i,s in enumerate(steps):
        op=s['op']
        if op=='predict':
            value=await controller.predict(s['robot'],s['component'],observation(i));tickets[s['as']]=value.get('ticket_id')
        elif op=='apply':value=await controller.apply(tickets[s['ticket']],execute_action)
        elif op=='join':value=await controller.join(s['robot'])
        elif op=='leave':value=await controller.leave(s['robot'])
        elif op=='fault':
            await asyncio.to_thread(transport.fail_worker,s['worker']);value=dict(status='injected')
            controller.emit('process_fault_injected',worker_id=s['worker'],kind_of_failure='owned Serve application deletion; no physical GPU failure')
        elif op=='failover':value=await controller.failover(s['worker'])
        elif op=='health':value=dict(status='inspected',workers=await controller.healthcheck())
        else:await asyncio.sleep(s['seconds']);value=dict(status='waited')
        row=dict(step=i,operation=op,status=value['status']);results.append(row);controller.emit('scenario_step',**row)
        if 'expect' in s and s['expect']!=value['status']:raise ValueError(f'Scenario step {i}: expected {s["expect"]}, got {value["status"]}')
    return results
