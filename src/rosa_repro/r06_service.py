"""Explicit regression batches over the production tensor adapter, not a performance server."""
from copy import deepcopy
import uuid
from .batching import SessionBook,predict_policy_batch
from .adapter import check_action_batch_one
from .regression import seeded_rng
from .paired import digest_value

class RegressionBatchWorker:
    def __init__(self,policy):self.policy=policy;self.book=SessionBook()
    def reset(self,run_id,robot_id,episode_id):return self.book.reset(run_id,robot_id,episode_id)
    def predict(self,items,arm):
        if arm not in ('official','shared_b1','selected') or not 1<=len(items)<=16:
            raise ValueError('Invalid regression batch')
        if arm!='selected' and len(items)!=1:raise ValueError('Reference arms require batch one')
        if any(set(i)!={'metadata','observation','seed'} or type(i['seed']) is not int or not 0<=i['seed']<2**32 for i in items):
            raise ValueError('Invalid request controls')
        tickets=[]
        try:
            for i in items:tickets.append(self.book.accept(i['metadata'],i['observation']))
            # The first row fixes the invocation seed. Across shapes this does NOT align per-row model noise.
            seed=items[0]['seed'];observations=[deepcopy(i['observation']) for i in items]
            with seeded_rng(seed):
                if arm=='official':
                    a,_=self.policy.get_action(observations[0]);check_action_batch_one(a);actions=[a]
                else:actions=predict_policy_batch(self.policy,observations)
            ident=uuid.uuid4().hex
            return [dict(metadata=deepcopy(i['metadata']),status='ok',action=a,batch_id=ident,
                actual_batch_size=len(items),batch_seed=seed,requested_seed=i['seed'],input_sha256=digest_value(i['observation']))
                for i,a in zip(items,actions)]
        finally:
            for ticket in tickets:self.book.finish(ticket)

class ZmqBatchTransport:
    """One REQ socket, owned and called sequentially by one executor thread."""
    def __init__(self,port,timeout_s):
        from gr00t.policy.server_client import PolicyClient
        self.client=PolicyClient(host='127.0.0.1',port=port,timeout_ms=round(timeout_s*1000))
    def reset(self,**kwargs):return self.client.call_endpoint('r06_reset',kwargs)
    def predict(self,items,arm):return self.client.call_endpoint('r06_predict_batch',dict(items=items,arm=arm))
    def evidence(self):return self.client.call_endpoint('r06_runtime',requires_input=False)
    def close(self):self.client.close()
