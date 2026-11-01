"""R02 batch-one shared worker and client guards. No model transforms here.

NumPy is the only non-stdlib dependency. Importing this module does not import
OR-Tools/jsonschema or alter the existing policy environment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time
import uuid

import numpy as np


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class Request:
    run_id: str
    robot_id: str
    episode_id: str
    request_id: str
    attempt_id: str
    component_id: str
    observation_seq: int
    generation: int

    def __post_init__(self):
        for name in ("run_id", "robot_id", "episode_id", "request_id", "attempt_id", "component_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ProtocolError(f"Invalid {name}")
        for name in ("observation_seq", "generation"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ProtocolError(f"Invalid {name}")
        if self.component_id != "s1":
            raise ProtocolError("R02 implements S1 only")


def check_observation_batch_one(observation):
    if not isinstance(observation, dict) or not observation:
        raise ProtocolError("Missing observation")
    for key, value in observation.items():
        if key.startswith(("video.", "state.")):
            if not isinstance(value, np.ndarray) or value.shape[0:1] != (1,):
                raise ProtocolError(f"R02 requires batch one: {key}")
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ProtocolError(f"Non-finite/non-numeric observation: {key}")
        else:
            # Preserve the official wrapper's accepted language containers.
            if isinstance(value, str):
                continue
            if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 1:
                raise ProtocolError(f"R02 requires one language item: {key}")
            if not isinstance(value[0], str):
                raise ProtocolError(f"Invalid language item: {key}")


def check_action_batch_one(action):
    if not isinstance(action, dict) or not action:
        raise ProtocolError("Missing action")
    horizons = set()
    for key, value in action.items():
        if (not key.startswith("action.") or not isinstance(value, np.ndarray)
                or value.dtype != np.float32 or value.ndim != 3
                or value.shape[0] != 1 or min(value.shape) < 1
                or not np.isfinite(value).all()):
            raise ProtocolError(f"Invalid/non-finite action: {key}")
        horizons.add(value.shape[1])
    if len(horizons) != 1:
        raise ProtocolError("Mixed action horizons")


class BatchOneWorker:
    """One loaded, explicitly stateless policy shared by robot sessions.

    Calls the public official get_action unchanged under a model lock. This is
    serialized batch-one serving, NOT dynamic batching or a throughput claim.
    Reset invalidates only the named session; it never resets another robot's
    model state. Stateful policies must use a different backend contract.
    """
    def __init__(self, policy, *, stateless_verified=False):
        if not stateless_verified:
            raise ProtocolError("Explicitly verify backend statelessness first")
        self.policy = policy
        self._state_lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._sessions = {}
        self._episodes = set()
        self._requests = set()

    def reset(self, run_id, robot_id, episode_id):
        Request(run_id, robot_id, episode_id, "reset", "reset", "s1", 0, 0)
        key = (run_id, robot_id)
        with self._state_lock:
            episode = (*key, episode_id)
            if episode in self._episodes:
                raise ProtocolError("Episode IDs may not be reused")
            old = self._sessions.get(key)
            generation = 0 if old is None else old["generation"] + 1
            self._sessions[key] = dict(episode_id=episode_id, generation=generation,
                                       last_seq=-1, busy=False)
            self._episodes.add(episode)
            return dict(run_id=run_id, robot_id=robot_id, episode_id=episode_id,
                        generation=generation)

    def predict(self, metadata, observation):
        req = Request(**metadata)
        check_observation_batch_one(observation)
        key = (req.run_id, req.robot_id)
        with self._state_lock:
            state = self._sessions.get(key)
            if (state is None or state["episode_id"] != req.episode_id
                    or state["generation"] != req.generation):
                raise ProtocolError("Stale or unknown episode")
            if (req.run_id, req.request_id) in self._requests:
                raise ProtocolError("Duplicate request ID")
            if state["busy"] or req.observation_seq <= state["last_seq"]:
                raise ProtocolError("Concurrent or out-of-order robot request")
            self._requests.add((req.run_id, req.request_id))
            state["last_seq"] = req.observation_seq
            state["busy"] = True
        response = dict(metadata=asdict(req), status="error", action=None,
                        batch_id=None, actual_batch_size=None)
        try:
            with self._model_lock:
                with self._state_lock:
                    if self._sessions[key] is not state:
                        response["status"] = "cancelled"
                        return response
                response.update(batch_id=uuid.uuid4().hex, actual_batch_size=1)
                # Public API retains the official input/output validation,
                # processor, collation, normalization, precision and RNG use.
                action, _ = self.policy.get_action(observation)
                check_action_batch_one(action)
            with self._state_lock:
                if self._sessions[key] is not state:
                    response["status"] = "cancelled"
                else:
                    response.update(status="ok", action=action)
            return response
        finally:
            with self._state_lock:
                state["busy"] = False


class JsonlSink:
    def __init__(self, path):
        self.file = Path(path).open("x", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, record):
        line = json.dumps(record, allow_nan=False, ensure_ascii=False)
        with self.lock:
            self.file.write(line + "\n")
            self.file.flush()

    def close(self):
        self.file.close()


class MeasuredClient:
    """R00 request telemetry and a once-only action gate in the client clock.

    Transport is injected, so tests and the official ZeroMQ connection use the
    same checks. No actions are invented from request counts. apply() holds the
    episode lock through execution; a reset cannot split an action chunk.
    """
    def __init__(self, *, run_id, robot_id, reset_call, predict_call, sink,
                 clock_domain, clock=time.monotonic_ns, deadline_ms=200, max_batch_size=1):
        if not math.isfinite(deadline_ms) or deadline_ms <= 0:
            raise ProtocolError("Invalid deadline")
        if not clock_domain:
            raise ProtocolError("Missing clock domain")
        if type(max_batch_size) is not int or not 1 <= max_batch_size <= 1024:
            raise ProtocolError("Invalid maximum batch size")
        self.max_batch_size = max_batch_size
        self.run_id, self.robot_id = run_id, robot_id
        self.reset_call, self.predict_call, self.sink = reset_call, predict_call, sink
        self.clock_domain, self.clock, self.deadline_ms = clock_domain, clock, deadline_ms
        self.lock = threading.RLock()
        self.session = None
        self.seq = 0
        self.pending = {}
        self.busy = False

    def reset(self, episode_id):
        with self.lock:
            ack = self.reset_call(self.run_id, self.robot_id, episode_id)
            if any(ack.get(k) != v for k, v in
                   dict(run_id=self.run_id, robot_id=self.robot_id, episode_id=episode_id).items()):
                raise ProtocolError("Reset acknowledgement identity mismatch")
            Request(self.run_id, self.robot_id, episode_id, "reset", "reset", "s1", 0, ack["generation"])
            self.session = dict(ack)
            self.seq = 0
            self.pending.clear()

    def predict(self, observation, *, warmup=False, scheduled_ns=None):
        with self.lock:
            if self.session is None or self.busy or self.pending:
                raise ProtocolError("Reset required or previous request/action still pending")
            if type(warmup) is not bool:
                raise ProtocolError("warmup must be boolean")
            req = Request(self.run_id, self.robot_id, self.session["episode_id"],
                          uuid.uuid4().hex, uuid.uuid4().hex, "s1", self.seq,
                          self.session["generation"])
            sent = self.clock()
            scheduled = sent if scheduled_ns is None else scheduled_ns
            if type(scheduled) is not int or not 0 <= scheduled <= sent:
                raise ProtocolError("Invalid scheduled time")
            self.seq += 1
            self.busy = True
        record = dict(schema_version="1.0", record_type="request",
                      **{k:v for k,v in asdict(req).items() if k != "generation"},
                      clock_domain=self.clock_domain, scheduled_ns=scheduled,
                      sent_ns=sent, finished_ns=sent, deadline_ms=self.deadline_ms,
                      status="error", valid_output=False, warmup=warmup,
                      batch_id=None, actual_batch_size=None)
        try:
            response = self.predict_call(asdict(req), observation)
            if response.get("metadata") != asdict(req):
                raise ProtocolError("Response identity mismatch")
            if response.get("status") not in ("ok", "cancelled", "rejected", "error", "timeout"):
                raise ProtocolError("Invalid response status")
            batch_id, size = response.get("batch_id"), response.get("actual_batch_size")
            if (batch_id is None) != (size is None) or (size is not None and
                    (type(size) is not int or not 1 <= size <= self.max_batch_size or not isinstance(batch_id, str) or not batch_id)):
                raise ProtocolError("Invalid batch metadata")
            record.update(status=response["status"], batch_id=batch_id, actual_batch_size=size)
            with self.lock:
                if (self.session["episode_id"] != req.episode_id
                        or self.session["generation"] != req.generation):
                    record["status"] = "cancelled"
                elif record["status"] == "ok":
                    if size is None:
                        raise ProtocolError("Successful inference lacks batch evidence")
                    check_action_batch_one(response["action"])
                    record["valid_output"] = True
                    # Keep our own copy: caller cannot substitute a returned action.
                    self.pending[req.request_id] = (req, {k:v.copy() for k,v in response["action"].items()})
            return req.request_id if record["valid_output"] else None
        except TimeoutError:
            record.update(status="timeout", valid_output=False)
            raise
        except Exception:
            record.update(status="error", valid_output=False)
            raise
        finally:
            record["finished_ns"] = self.clock()
            with self.lock:
                try:
                    self.sink(record)
                except Exception:
                    # An unrecorded response must never become executable.
                    self.pending.pop(req.request_id, None)
                    raise
                finally:
                    self.busy = False

    def apply(self, request_id, execute):
        """execute(action) returns actual integer number of executed sim steps.

        If execution raises, the run must be marked failed: partial application
        is unknown and no fabricated action record is emitted. Token is consumed.
        """
        with self.lock:
            if self.busy or request_id not in self.pending:
                raise ProtocolError("Stale, unknown or already consumed response")
            req, action = self.pending.pop(request_id)
            start = self.clock()
            steps = execute(action)
            if type(steps) is not int or not 0 <= steps <= next(iter(action.values())).shape[1]:
                raise ProtocolError("Executor must report actual steps within action horizon")
            self.sink(dict(schema_version="1.0", record_type="action",
                           **{k:getattr(req,k) for k in
                              ("run_id", "robot_id", "episode_id", "request_id", "component_id")},
                           clock_domain=self.clock_domain, action_id=uuid.uuid4().hex,
                           start_ns=start, end_ns=self.clock(), applied=steps>0,
                           executed_steps=steps))
            return steps
