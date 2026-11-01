"""ROSA §3.4.3: adaptive frontier, isolated ILP packing, greedy compaction.

Always returns independently checked feasible witnesses. Heuristic rejection
is NOT a proof of infeasibility. Search brackets are heuristic, not certified
optimality bounds. Homogeneous physical GPUs; heterogeneous task classes.
"""
from dataclasses import dataclass,asdict
import math

from .contracts import obj,TEXT,NUMBER,POSITIVE,POSINT,COMPONENT,PROFILE,validate
from .domain import Problem

TASK=obj({'id':TEXT,'robots':{'type':'integer','minimum':0,'maximum':64},'weight':NUMBER,
          'min_action_rate_hz':NUMBER,'action_time_s':POSITIVE,'s2_horizon':POSINT,
          'components':{'type':'array','minItems':1,'items':COMPONENT}})
HETERO_SCHEMA={'$schema':'https://json-schema.org/draft/2020-12/schema',**obj({
    'schema_version':{'const':'1.0'},'id':TEXT,
    'classes':{'type':'array','minItems':1,'maxItems':8,'items':TASK},
    'hardware':obj({'kind':TEXT,'gpu_count':{'type':'integer','minimum':0,'maximum':128},'memory_mib':POSITIVE}),
    'profiles':{'type':'array','items':PROFILE},
    'search':obj({'objective_tolerance':POSITIVE,'time_limit_s':NUMBER,'max_iterations':{'type':'integer','minimum':1,'maximum':4096}})})}


@dataclass
class Fleet:
    raw: dict
    problems: dict
    def __post_init__(self):
        self.classes={c['id']:c for c in self.raw['classes']}
        self.profiles={p.id:p for problem in self.problems.values() for p in problem.profiles}
        self.hardware=self.raw['hardware']
    @classmethod
    def from_dict(cls,data):
        validate(data,HETERO_SCHEMA)
        classes=data['classes']
        if len({c['id'] for c in classes})!=len(classes):raise ValueError('Duplicate task class')
        identities={};required=set();problems={}
        for c in classes:
            if c['robots']==0 and c['min_action_rate_hz']>0:raise ValueError('Empty class has positive Fmin')
            for model in c['components']:
                required.add(model['id'])
                fingerprint=(model['role'],model['model_revision'],model['runtime_fingerprint'])
                if model['id'] in identities and identities[model['id']]!=fingerprint:
                    raise ValueError('Shared component identity has incompatible role/revision/runtime')
                identities[model['id']]=fingerprint
            ids={m['id'] for m in c['components']}
            p=dict(schema_version='1.0',workload={k:c[k] for k in ('id','robots','action_time_s','s2_horizon','components')},
                hardware=data['hardware'],profiles=[p for p in data['profiles'] if p['component_id'] in ids],
                search={'frequency_tolerance_hz':.01,'time_limit_s':data['search']['time_limit_s']})
            problems[c['id']]=Problem.from_dict(p)
        if any(p['component_id'] not in required for p in data['profiles']):raise ValueError('Profile has no declared model component')
        if len({p['id'] for p in data['profiles']})!=len(data['profiles']):raise ValueError('Duplicate global profile ID')
        return cls(data,problems)
    def rate(self,cid,component_id,frequencies):
        p=self.problems[cid];c=p.component(component_id)
        return c.periodic_hz if c.role=='obligation' else frequencies[cid]/(p.s2_horizon if c.role=='s2' else 1)
    def objective(self,f):return sum(c['weight']*c['robots']*f[cid] for cid,c in self.classes.items())


def member_key(m):return m['class_id'],m['robot_id']


def routes_for(fleet,workers):
    routes={cid:{str(i):{} for i in range(c['robots'])} for cid,c in fleet.classes.items()}
    for index,w in enumerate(workers):
        w['worker_id']=f'gpu-{index}'
        for m in w['members']:routes[m['class_id']][m['robot_id']][w['component_id']]=w['worker_id']
    return routes


def validate_fleet_schedule(fleet,frequencies,workers,routes,*,check_gpu_budget=True):
    """Recompute coverage, profile/SLO, rates, memory, Fmin and per-robot loops."""
    if set(frequencies)!=set(fleet.classes):raise ValueError('Frequency class mismatch')
    for cid,c in fleet.classes.items():
        f=frequencies[cid]
        if not math.isfinite(f) or f<0 or f>1/c['action_time_s']+1e-12:raise ValueError('Invalid class rate')
        if not c['robots'] and f!=0:raise ValueError('Empty class rate must be zero')
        if c['robots']*f+1e-12<c['min_action_rate_hz']:raise ValueError('Fmin violated')
    if check_gpu_budget and len(workers)>fleet.hardware['gpu_count']:raise ValueError('GPU budget exceeded')
    by_worker={w['worker_id']:w for w in workers}
    if len(by_worker)!=len(workers):raise ValueError('Duplicate GPU worker')
    seen=set()
    for w in workers:
        p=fleet.profiles.get(w['profile_id'])
        if p is None or p.component_id!=w['component_id']:raise ValueError('Wrong model/profile on worker')
        if not w['members']:raise ValueError('Empty worker')
        rate=0;deadlines=[]
        for m in w['members']:
            cid,rid=member_key(m)
            if cid not in fleet.classes or rid not in {str(i) for i in range(fleet.classes[cid]['robots'])}:raise ValueError('Unknown class/robot')
            comp=next((c for c in fleet.problems[cid].components if c.id==w['component_id']),None)
            if comp is None:raise ValueError('Unneeded component assigned to robot')
            key=(cid,rid,comp.id)
            if key in seen:raise ValueError('Robot component assigned twice')
            seen.add(key);rate+=fleet.rate(cid,comp.id,frequencies);deadlines.append(comp.deadline_ms)
            if routes.get(cid,{}).get(rid,{}).get(comp.id)!=w['worker_id']:raise ValueError('Forward route mismatch')
        est=p.at(rate)
        if est.status!='feasible' or est.p99_ms>min(deadlines):raise ValueError('Shared worker violates a class SLO/profile range')
        if p.memory_mib>fleet.hardware['memory_mib']:raise ValueError('Worker memory exceeded')
        expected=(rate,est.mean_ms,est.p99_ms,p.batch_size,p.batch_wait_ms,p.memory_mib)
        actual=tuple(w[k] for k in ('rate_hz','mean_ms','p99_ms','batch_size','batch_wait_ms','memory_mib'))
        if any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(actual,expected)):
            raise ValueError('Worker profile values/rates changed')
    if set(routes)!=set(fleet.classes):raise ValueError('Route classes mismatch')
    for cid,c in fleet.classes.items():
        if set(routes[cid])!={str(i) for i in range(c['robots'])}:raise ValueError('Route robot coverage mismatch')
        p=fleet.problems[cid]
        for rid,mapping in routes[cid].items():
            if set(mapping)!={comp.id for comp in p.components}:raise ValueError('Missing required component route')
            loop=p.action_time_s
            for comp in p.components:
                w=by_worker.get(mapping[comp.id])
                if w is None or w['component_id']!=comp.id or (cid,rid,comp.id) not in seen or not any(member_key(m)==(cid,rid) for m in w['members']):
                    raise ValueError('Reverse route mismatch')
                if comp.role!='obligation':loop+=w['mean_ms']/1000/(p.s2_horizon if comp.role=='s2' else 1)
            if frequencies[cid]*loop>1+1e-12:raise ValueError('Closed-loop rate exceeded')


class HeterogeneousScheduler:
    def __init__(self,fleet):self.fleet=fleet
    def _isolated(self,frequencies,role,budget):
        from .scheduler import enumerate_configurations,pack
        workers=[];unknown=[]
        for cid,p in self.fleet.problems.items():
            if not p.robots:continue
            for comp in p.components:
                if (comp.role=='obligation')!=(role=='obligation'):continue
                configs,unresolved=enumerate_configurations(p,[comp],frequencies[cid],budget)
                unknown+=unresolved
                packed=pack(configs,[comp],p.robots,p.robots,budget,obligations=role=='obligation')
                if packed.status!='feasible':
                    return None,unknown,packed.reason
                cursor=0
                for conf in packed.selected:
                    members=[{'class_id':cid,'robot_id':str(i)} for i in range(cursor,cursor+conf.streams)]
                    cursor+=conf.streams
                    w=asdict(conf);w.pop('streams');w['members']=members
                    workers.append(w)
        return workers,unknown,''
    def _headroom(self,w):
        p=self.fleet.profiles[w['profile_id']]
        deadline=min(self.fleet.problems[m['class_id']].component(w['component_id']).deadline_ms for m in w['members'])
        rates=[x.rate_hz for x in p.points if x.p99_ms<=deadline]
        return max(rates,default=w['rate_hz'])-w['rate_hz']
    def _merge_candidate(self,a,b,frequencies):
        if a['component_id']!=b['component_id']:return None,False
        members=a['members']+b['members']
        if len({m['class_id'] for m in members})<2:return None,False
        component=a['component_id'];rate=sum(self.fleet.rate(m['class_id'],component,frequencies) for m in members)
        deadline=min(self.fleet.problems[m['class_id']].component(component).deadline_ms for m in members)
        options=[];unknown=False
        for p in self.fleet.profiles.values():
            if p.component_id!=component or p.memory_mib>self.fleet.hardware['memory_mib']:continue
            e=p.at(rate)
            unknown|=e.status=='unknown'
            if e.status=='feasible' and e.p99_ms<=deadline:
                options.append(dict(component_id=component,profile_id=p.id,members=members,
                    rate_hz=rate,mean_ms=e.mean_ms,p99_ms=e.p99_ms,batch_size=p.batch_size,
                    batch_wait_ms=p.batch_wait_ms,memory_mib=p.memory_mib))
        options.sort(key=lambda w:(w['mean_ms'],w['batch_size'],w['profile_id']))
        return options,unknown
    def _compact(self,workers,frequencies,budget,*,other=(),goals=False):
        workers=list(workers);trace=[];unknown=False
        while budget.remaining():
            merged=False
            sources=sorted(range(len(workers)),key=lambda i:(workers[i]['rate_hz'],workers[i]['component_id'],i))
            for i in sources:
                targets=sorted((j for j in range(len(workers)) if j!=i),key=lambda j:(-self._headroom(workers[j]),j))
                for j in targets:
                    if not budget.remaining():return workers,trace,True
                    options,unresolved=self._merge_candidate(workers[i],workers[j],frequencies);unknown|=unresolved
                    for candidate in options or []:
                        proposed=[w for k,w in enumerate(workers) if k not in (i,j)]+[candidate]
                        if goals:
                            complete=list(other)+proposed;routes=routes_for(self.fleet,complete)
                            try:validate_fleet_schedule(self.fleet,frequencies,complete,routes,check_gpu_budget=False)
                            except ValueError as exc:
                                if str(exc)=='Closed-loop rate exceeded':continue
                                raise
                        trace.append(dict(component_id=candidate['component_id'],combined_rate_hz=candidate['rate_hz'],
                            class_ids=sorted({m['class_id'] for m in candidate['members']}),
                            profile_id=candidate['profile_id'],workers_before=len(workers),workers_after=len(proposed)))
                        workers=proposed;merged=True;break
                    if merged:break
                if merged:break
            if not merged:break
        return workers,trace,unknown
    def provision(self,budget):
        zero={cid:0. for cid in self.fleet.classes}
        isolated,unknown,reason=self._isolated(zero,'obligation',budget)
        if isolated is None:return {'status':'unknown','category':'profile_or_solver_unknown','reason':reason}
        compacted,merges,unresolved=self._compact(isolated,zero,budget)
        if not budget.remaining():return {'status':'unknown','category':'time_budget','reason':'obligation_time_budget'}
        return dict(status='feasible',workers=compacted,isolated_workers=len(isolated),merges=merges,
                    minimum_footprint_proven=False,profile_unknown=bool(unknown) or unresolved)
    def check_rates(self,frequencies,budget=None,provisioned=None):
        result=self._check_rates(frequencies,budget,provisioned)
        return dict(schema_version='1.0',algorithm='rosa-v1-heterogeneous-independent',
                    gpu_execution=False,profile_sources=sorted({p.source for p in self.fleet.profiles.values()}),
                    **result)
    def _check_rates(self,frequencies,budget=None,provisioned=None):
        from .scheduler import Budget
        fleet=self.fleet;budget=budget or Budget(fleet.raw['search']['time_limit_s'])
        if set(frequencies)!=set(fleet.classes):raise ValueError('Frequency class mismatch')
        for cid,c in fleet.classes.items():
            f=frequencies[cid]
            if not math.isfinite(f) or f<0:raise ValueError('Invalid frequency')
            if c['robots']==0 and f!=0:raise ValueError('Empty class requires zero rate')
            if f>1/c['action_time_s']+1e-12 or f*c['robots']+1e-12<c['min_action_rate_hz']:
                return dict(status='infeasible',category='necessary_bound',reason='action_time_or_Fmin_bound')
        required={comp.id for cid,p in fleet.problems.items() if p.robots for comp in p.components}
        if len(required)>fleet.hardware['gpu_count']:
            return dict(status='infeasible',category='necessary_bound',reason='one_gpu_per_required_component')
        if not budget.remaining():return dict(status='unknown',category='time_budget',reason='time_budget')
        obligation=provisioned if provisioned is not None else self.provision(budget)
        if obligation['status']!='feasible':return obligation
        goals,unknown,reason=self._isolated(frequencies,'goal',budget)
        if goals is None:
            return dict(status='unknown',category='profile_or_solver_unknown' if unknown or not budget.remaining() else 'heuristic_rejection',reason=reason)
        before=len(goals);merges=[];unresolved=False
        if len(goals)+len(obligation['workers'])>fleet.hardware['gpu_count']:
            goals,merges,unresolved=self._compact(goals,frequencies,budget,other=obligation['workers'],goals=True)
        workers=list(obligation['workers'])+goals;routes=routes_for(fleet,workers)
        try:validate_fleet_schedule(fleet,frequencies,workers,routes)
        except ValueError as exc:
            if str(exc) not in ('GPU budget exceeded','Closed-loop rate exceeded'):raise
            return dict(status='unknown',category='profile_or_solver_unknown' if unknown or unresolved else 'heuristic_rejection',
                        reason=str(exc),isolated_goal_workers=before,compacted_goal_workers=len(goals))
        return dict(status='feasible',frequencies_hz=dict(frequencies),weighted_objective=fleet.objective(frequencies),
            placements=workers,routes=routes,isolated_goal_workers=before,compacted_goal_workers=len(goals),
            goal_merges=merges,obligations=obligation,unknown_profile_queries=len(unknown),optimality_proven=False)
    def solve(self):
        from .scheduler import Budget
        fleet=self.fleet;search=fleet.raw['search'];budget=Budget(search['time_limit_s'])
        base=dict(schema_version='1.0',algorithm='rosa-v1-heterogeneous-independent',gpu_execution=False,
            profile_sources=sorted({p.source for p in fleet.profiles.values()}),trace=[],optimality_proven=False)
        if any(not p.monotone for p in fleet.profiles.values()):
            return dict(base,status='unknown',reason='adaptive_search_requires_monotone_profiles')
        lo={cid:c['min_action_rate_hz']/c['robots'] if c['robots'] else 0. for cid,c in fleet.classes.items()}
        hi={cid:1/c['action_time_s'] if c['robots'] else 0. for cid,c in fleet.classes.items()}
        if any(lo[c]>hi[c] for c in lo):return dict(base,status='infeasible',reason='Fmin_exceeds_action_time_bound')
        obligation=self.provision(budget)
        best=self.check_rates(lo,budget,obligation)
        if best['status']!='feasible':return {**base,**best,'reason':'initial_frontier:'+best.get('reason','')}
        blocked=set();pruned=False;termination='iteration_limit'
        for _ in range(search['max_iterations']):
            if not budget.remaining():termination='time_budget';break
            gains={cid:fleet.classes[cid]['weight']*fleet.classes[cid]['robots']*(hi[cid]-lo[cid]) for cid in lo if cid not in blocked}
            if not gains or sum(gains.values())<=search['objective_tolerance']:
                termination='unknown_dimensions' if blocked else 'heuristic_tolerance';break
            cid=max(sorted(gains),key=lambda c:gains[c])
            candidate=dict(lo);candidate[cid]=(lo[cid]+hi[cid])/2
            result=self.check_rates(candidate,budget,obligation)
            base['trace'].append(dict(class_id=cid,frequencies_hz=candidate,status=result['status'],
                category=result.get('category'),reason=result.get('reason','')))
            if result['status']=='feasible':lo=candidate;best=result
            elif result.get('category') in ('necessary_bound','heuristic_rejection'):
                hi[cid]=candidate[cid]
                pruned|=result.get('category')=='heuristic_rejection'
            else:blocked.add(cid)
        # Re-check the final witness because internal compaction reuses lists.
        validate_fleet_schedule(fleet,best['frequencies_hz'],best['placements'],best['routes'])
        return {**base,**best,'search_status':termination,'solution_quality':'validated_heuristic_lower_bound',
            'heuristic_pruning_used':pruned,'unresolved_dimensions':sorted(blocked),
            'heuristic_rate_brackets_hz':{cid:[lo[cid],hi[cid]] for cid in lo},
            'objective_absolute_upper_bound':sum(c['weight']*c['robots']/c['action_time_s'] for c in fleet.classes.values()),
            'minimum_obligation_footprint_proven':False}
