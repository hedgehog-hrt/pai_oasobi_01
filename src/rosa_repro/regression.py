"""Saved-observation and paired-RNG regression helpers for pinned GR00T."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import uuid

import numpy as np

from .adapter import BatchOneWorker, Request, check_action_batch_one, check_observation_batch_one

UPSTREAM_COMMIT = "944fe70c4fd54e3fcb80a7d7258a328f9c061176"
MODEL_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
T01_CONFIG_HASHES = {
    "config.json": "ce763f74ad1cda2ca9e67491229cbc5810d83cae675cb4dad839a69a9731fd97",
    "model.safetensors.index.json": "407804ea5a62f4f8823f48811ae0edbb82fac101e9cf4d7273e6e2f692bb4d59",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def save_observation(directory, observation, provenance):
    """No pickle, lossy casts or image resizing. Array bytes live in NPZ."""
    check_observation_batch_one(observation)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    arrays, fields = {}, {}
    for index, (key, value) in enumerate(sorted(observation.items())):
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                # Official language ndarray may have object dtype. Normalize
                # only its container to list; numeric observations stay exact.
                if key.startswith(("state.", "video.")):
                    raise ValueError("Object observation arrays are not supported")
                fields[key] = dict(kind="list", value=value.tolist())
            else:
                name = f"a{index}"
                arrays[name] = value
                fields[key] = dict(kind="array", name=name, dtype=str(value.dtype), shape=list(value.shape))
        elif isinstance(value, (str, list, tuple)):
            fields[key] = dict(kind=type(value).__name__, value=value)
        else:
            raise TypeError(f"Unsupported observation value: {key}")
    np.savez_compressed(directory / "arrays.npz", **arrays)
    manifest = dict(schema_version="1.0", fields=fields, provenance=provenance,
                    arrays_sha256=sha256(directory/"arrays.npz"))
    write_json(directory/"observation.json", manifest)
    return manifest


def load_observation(directory):
    directory = Path(directory)
    manifest = json.loads((directory/"observation.json").read_text())
    if manifest["schema_version"] != "1.0" or sha256(directory/"arrays.npz") != manifest["arrays_sha256"]:
        raise ValueError("Observation version/hash mismatch")
    obs = {}
    with np.load(directory/"arrays.npz", allow_pickle=False) as arrays:
        for key, field in manifest["fields"].items():
            kind = field["kind"]
            if kind == "array":
                value = arrays[field["name"]]
                if str(value.dtype) != field["dtype"] or list(value.shape) != field["shape"]:
                    raise ValueError("Observation schema mismatch")
                obs[key] = value
            elif kind in ("str", "list", "tuple"):
                obs[key] = tuple(field["value"]) if kind == "tuple" else field["value"]
            else:
                raise ValueError("Unsupported observation encoding")
    check_observation_batch_one(obs)
    return obs, manifest


def copy_observation(observation):
    import copy
    return copy.deepcopy(observation)


def compare_actions(reference, candidate, *, atol=0.0, rtol=0.0):
    if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
        raise ValueError("Invalid comparison tolerance")
    check_action_batch_one(reference)
    check_action_batch_one(candidate)
    if set(reference) != set(candidate):
        return dict(passed=False, reason="action_keys_differ", fields={})
    fields = {}
    for key in sorted(reference):
        a, b = reference[key], candidate[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            fields[key] = dict(passed=False, reason="shape_or_dtype_differs")
        else:
            error = np.abs(a.astype(np.float64) - b.astype(np.float64))
            fields[key] = dict(passed=bool(np.allclose(a,b,atol=atol,rtol=rtol)),
                               exact_equal=bool(np.array_equal(a,b)),
                               max_abs_error=float(error.max()),
                               mean_abs_error=float(error.mean()), shape=list(a.shape), dtype=str(a.dtype))
    return dict(passed=all(f["passed"] for f in fields.values()), atol=atol, rtol=rtol, fields=fields)


@contextmanager
def seeded_rng(seed):
    """Restore all default Python/NumPy/Torch CPU/CUDA RNG states on exit.

    Does not toggle deterministic algorithms/TF32/compile or alter official
    kernels. Equality is measured, not assumed. Dedicated Generators, if an
    upstream revision introduces them, require a new contract/review.
    """
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_states is not None:
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def paired_compare(policy, observation, *, seed, atol=0.0, rtol=0.0, include_actions=False):
    """Same loaded stateless model, same observation, restored default RNGs.

    Run this on an idle worker only. This isolates adapter effects; it does not
    establish multi-process/server or hardware reproducibility.
    """
    check_observation_batch_one(observation)
    with seeded_rng(seed):
        reference, _ = policy.get_action(copy_observation(observation))
    worker = BatchOneWorker(policy, stateless_verified=True)
    run_id = "regression-"+uuid.uuid4().hex
    ack = worker.reset(run_id, "robot0", "episode0")
    req = Request(run_id, "robot0", "episode0", "request0", "attempt0", "s1", 0, ack["generation"])
    with seeded_rng(seed):
        response = worker.predict(asdict(req), copy_observation(observation))
    if response["metadata"] != asdict(req) or response["status"] != "ok":
        raise ValueError("Regression response identity/status mismatch")
    result = compare_actions(reference, response["action"], atol=atol, rtol=rtol)
    result.update(seed=seed, batch_size=1, same_loaded_model=True,
                  default_rngs_restored=True, compared_path="official_wrapper_vs_batch_one_worker")
    if include_actions:result.update(reference_actions=copy_observation(reference),candidate_actions=copy_observation(response['action']))
    return result


def verify_upstream(repo):
    repo = Path(repo)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Unexpected GR00T commit: {commit}")
    relevant = ["gr00t/policy/policy.py", "gr00t/policy/gr00t_policy.py", "gr00t/policy/server_client.py"]
    # Untracked runtime aliases are allowed; tracked inference edits are not.
    tracked_diff = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--", "gr00t"])
    if tracked_diff:
        raise ValueError("Tracked upstream inference files changed; review before regression")
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True)
    return dict(commit=commit, tracked_gr00t_diff_empty=True, git_status=status,
                policy_sources_sha256={p:sha256(repo/p) for p in relevant})


def verify_checkpoint_config(model_path):
    actual = {name:sha256(Path(model_path)/name) for name in T01_CONFIG_HASHES}
    if actual != T01_CONFIG_HASHES:
        raise ValueError("Model configuration/index differs from the completed T01 baseline")
    return dict(expected_revision=MODEL_REVISION, config_sha256=actual,
                full_weight_hashes_verified=False)


def runtime_evidence(policy):
    """Observations only: absence of compile markers is not proof of no graphs."""
    import torch
    from .runtime_attestation import torch_state,model_state
    model = policy.policy.model
    dtypes = Counter(str(p.dtype) for p in model.parameters())
    compiled, attention = [], {}
    for name, module in model.named_modules():
        if hasattr(module, "_orig_mod"):
            compiled.append(name)
        config = getattr(module, "config", None)
        backend = getattr(config, "_attn_implementation", None)
        if backend is not None:
            attention[name] = str(backend)
    return dict(torch_version=torch.__version__, model_class=type(model).__name__,
                observed_torch_state=torch_state(torch),observed_model_state=model_state(model),
                parameter_tensor_counts_by_dtype=dict(dtypes), compiled_module_markers=compiled,
                attention_config_values=attention, cuda_graphs=None,
                unknown_fields=["cuda_graphs", "complete_kernel_level_attention_backend"],
                note="Parameter dtype/config/compile markers observed; kernel trace not collected.")
