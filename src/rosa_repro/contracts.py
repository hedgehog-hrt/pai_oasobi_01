"""Versioned strict JSON contracts; finite numbers are checked separately."""
from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required,
            "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1}
NUMBER = {"type": "number", "minimum": 0, "maximum": 1e9}
POSITIVE = {"type": "number", "exclusiveMinimum": 0, "maximum": 1e9}
INTEGER = {"type": "integer", "minimum": 0}
POSINT = {"type": "integer", "minimum": 1}
SHA256 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}

COMPONENT = obj({
    "id": TEXT, "role": {"enum": ["s1", "s2", "obligation"]},
    "model_revision": TEXT, "runtime_fingerprint": TEXT,
    "deadline_ms": POSITIVE, "periodic_hz": POSITIVE,
}, ["id", "role", "model_revision", "runtime_fingerprint", "deadline_ms"])

PROFILE = obj({
    "id": TEXT, "component_id": TEXT, "hardware_kind": TEXT,
    "model_revision": TEXT, "runtime_fingerprint": TEXT,
    "source": {"enum": ["synthetic", "measured"]},
    "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
    "batch_wait_ms": NUMBER, "memory_mib": POSITIVE,
    "capacity_bound_hz": POSITIVE,
    "evidence_sha256": SHA256,
    "points": {"type": "array", "minItems": 1, "items": obj({
        "rate_hz": NUMBER, "mean_ms": NUMBER, "p99_ms": NUMBER,
        "sample_count": POSINT, "duration_s": POSITIVE,
    }, ["rate_hz", "mean_ms", "p99_ms"])},
}, ["id", "component_id", "hardware_kind", "model_revision", "runtime_fingerprint",
    "source", "batch_size", "batch_wait_ms", "memory_mib", "points"])

PROBLEM_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **obj({
        "schema_version": {"const": "1.0"},
        "workload": obj({
            "id": TEXT, "robots": {"type": "integer", "minimum": 0, "maximum": 1024},
            "action_time_s": POSITIVE, "s2_horizon": POSINT,
            "components": {"type": "array", "minItems": 1, "items": COMPONENT},
        }),
        "hardware": obj({
            "kind": TEXT, "gpu_count": {"type": "integer", "minimum": 0, "maximum": 128},
            "memory_mib": POSITIVE,
        }),
        "profiles": {"type": "array", "items": PROFILE},
        "search": obj({"frequency_tolerance_hz": POSITIVE, "time_limit_s": NUMBER}),
    }),
}

IDENTITY = {"run_id": TEXT, "robot_id": TEXT, "episode_id": TEXT,
            "request_id": TEXT, "component_id": TEXT}
REQUEST_SCHEMA = obj({
    "schema_version": {"const": "1.0"}, "record_type": {"const": "request"},
    **IDENTITY, "attempt_id": TEXT, "observation_seq": INTEGER,
    "clock_domain": TEXT, "scheduled_ns": INTEGER, "sent_ns": INTEGER,
    "finished_ns": INTEGER, "deadline_ms": POSITIVE,
    "status": {"enum": ["ok", "timeout", "rejected", "error", "cancelled"]},
    "valid_output": {"type": "boolean"}, "warmup": {"type": "boolean"},
    "batch_id": {"type": ["string", "null"]},
    "actual_batch_size": {"type": ["integer", "null"], "minimum": 1},
})
ACTION_SCHEMA = obj({
    "schema_version": {"const": "1.0"}, "record_type": {"const": "action"},
    **IDENTITY, "clock_domain": TEXT, "action_id": TEXT,
    "start_ns": INTEGER, "end_ns": INTEGER,
    "applied": {"type": "boolean"}, "executed_steps": INTEGER,
})
MEASUREMENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "oneOf": [REQUEST_SCHEMA, ACTION_SCHEMA],
}

RUN_MANIFEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    **obj({
        "schema_version": {"const": "1.0"}, "run_id": TEXT,
        "mode": {"enum": ["cpu_solver", "gpu_replay", "live_simulation"]},
        "measurement_phase": {"enum": ["exploratory", "formal"]},
        "source_sha256": SHA256,
        "git_commit": {"type": ["string", "null"]},
        "git_dirty": {"type": ["boolean", "null"]},
        "git_diff_sha256": {"type": ["string", "null"]},
        "dependency_lock_sha256": SHA256, "python_version": TEXT,
        "clock_domain": TEXT,
        "gpu_count": INTEGER,
        "gpus": {"type": "array", "items": obj({"uuid": TEXT, "kind": TEXT,
                       "memory_mib": POSITIVE, "driver": TEXT})},
        "model_runtime": {"type": "array", "items": obj({
            "component_id": TEXT, "revision": TEXT,
            "dtype": {"type": ["string", "null"]},
            "torch_compile": {"type": ["boolean", "null"]},
            "cuda_graphs": {"type": ["boolean", "null"]},
            "attention_backend": {"type": ["string", "null"]},
            "input_schema_sha256": {"type": ["string", "null"]},
        })},
        "initial_states_sha256": {"type": ["string", "null"]},
        "observation_manifest_sha256": {"type": ["string", "null"]},
        "profile_manifest_sha256": {"type": ["string", "null"]},
        "unknown_fields": {"type": "array", "items": TEXT, "uniqueItems": True},
    }),
}


def validate_manifest(data):
    validate(data, RUN_MANIFEST_SCHEMA)
    if data['gpu_count'] != len(data['gpus']):
        raise ValueError('GPU count and GPU inventory disagree')
    if len({g['uuid'] for g in data['gpus']}) != len(data['gpus']):
        raise ValueError('Duplicate physical GPU UUIDs')
    for key in ('git_diff_sha256','initial_states_sha256','observation_manifest_sha256','profile_manifest_sha256'):
        if data[key] is not None:
            validate(data[key], SHA256)
    if data['git_dirty'] is True and data['git_diff_sha256'] is None:
        raise ValueError('Dirty source must include a diff hash')
    if data['measurement_phase'] == 'formal':
        if data['unknown_fields']:
            raise ValueError('Formal manifest still has unknown fields')
        if data['mode'] != 'cpu_solver':
            if data['gpu_count'] == 0 or not data['model_runtime']:
                raise ValueError('GPU experiment lacks hardware/model records')
            for model in data['model_runtime']:
                if any(model[k] is None for k in ('dtype','torch_compile','cuda_graphs','attention_backend','input_schema_sha256')):
                    raise ValueError('Runtime precision/optimization must be observed for formal runs')
                validate(model['input_schema_sha256'], SHA256)
        if data['mode']=='live_simulation' and data['initial_states_sha256'] is None:
            raise ValueError('Formal simulation needs initial-state manifest')
        if data['mode']=='gpu_replay' and data['observation_manifest_sha256'] is None:
            raise ValueError('Formal replay needs an observation manifest')


def validate(data, schema):
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("NaN and Infinity are forbidden")
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)
    finite(data)
    Draft202012Validator(schema).validate(data)


def load_json(path):
    def reject(value):
        raise ValueError(f"Invalid JSON numeric constant: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), parse_constant=reject, object_pairs_hook=unique)
