"""Pinned official GR00T + R02 endpoints; existing policy environment only."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rosa_repro.adapter import BatchOneWorker, Request, check_action_batch_one
from rosa_repro.regression import (paired_compare, runtime_evidence, verify_upstream,write_json, seeded_rng)
from rosa_repro.checkpoint_verification import verify_groot_checkpoint
from rosa_repro.execution_plan import runtime_source_manifest
from rosa_repro.action_regression import collect_arrays
from rosa_repro.r06_service import RegressionBatchWorker


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=5556)
    args = p.parse_args()
    source = verify_upstream(args.repo)
    if not args.model_path.is_dir():
        p.error('Model checkpoint directory missing')
    checkpoint = verify_groot_checkpoint(args.model_path)
    if Path.cwd().resolve() != args.repo.resolve():
        p.error('Run from the GR00T repository so the pinned local Cosmos alias resolves')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repo.resolve()))
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    policy = Gr00tSimPolicyWrapper(Gr00tPolicy(
        embodiment_tag='LIBERO_PANDA', model_path=str(args.model_path), device='cuda:0', strict=True))
    worker = BatchOneWorker(policy, stateless_verified=True)
    r06_worker = RegressionBatchWorker(policy)
    def predict_seeded(metadata, observation, seed, arm):
        if arm not in ('official', 'shared') or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('Invalid paired-regression controls')
        Request(**metadata)
        with seeded_rng(seed):
            if arm == 'shared':
                return worker.predict(metadata, observation)
            # Official public API with regression-only metadata/RNG control.
            # This endpoint is not an unmodified official-server latency baseline.
            action, _ = policy.get_action(observation)
            check_action_batch_one(action)
            return dict(metadata=metadata, status='ok', action=action,
                        batch_id='official-'+metadata['request_id'], actual_batch_size=1)
    evidence = dict(source=source, checkpoint=checkpoint, runtime=runtime_evidence(policy),rosa_source_manifest=runtime_source_manifest(),
                    mode='r02-official-zmq-serialized-batch-one', model_path=str(args.model_path),
                    ray_serve=False, dynamic_batching=False, model_revision_verified=checkpoint['full_weight_hashes_verified'],
                    revision_verification_scope='pinned GR00T inference checkpoint; external runtime dependencies separate')
    configs=policy.get_modality_config()
    r06_evidence=dict(evidence, mode='r06-explicit-regression-batches',
        modality_config={k:dict(delta_indices=list(v.delta_indices), modality_keys=list(v.modality_keys)) for k,v in configs.items()})
    write_json(args.output/'r06_evidence.json',r06_evidence)
    write_json(args.output/'startup_evidence.json', evidence)
    with PolicyServer(policy, host='127.0.0.1', port=args.port) as server:
        server.register_endpoint('r06_reset', r06_worker.reset)
        server.register_endpoint('r06_predict_batch', r06_worker.predict)
        server.register_endpoint('r06_runtime', lambda:r06_evidence, requires_input=False)
        server.register_endpoint('r02_reset', worker.reset)
        server.register_endpoint('r02_predict', worker.predict)
        server.register_endpoint('r02_predict_seeded', predict_seeded)
        server.register_endpoint('r02_compare', lambda observation, seed, atol=0.0, rtol=0.0:
                                 paired_compare(policy, observation, seed=seed, atol=atol, rtol=rtol))
        server.register_endpoint('r02_runtime', lambda: evidence, requires_input=False)
        server.register_endpoint('r02_compare_arrays',lambda observations,seed,atol=0.,rtol=0.,comparison='same_shape':
                                 collect_arrays(policy,observations,seed=seed,atol=atol,rtol=rtol,comparison=comparison))
        server.run()


if __name__ == '__main__':
    main()
