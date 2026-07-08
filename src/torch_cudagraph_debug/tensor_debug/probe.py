"""Object API for CUDA Graph tensor probes."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch.utils.hooks import RemovableHandle

from ._collector import (
    DeviceLike,
    ProbeWhen,
    SynchronizeTarget,
    TensorAction,
    _TensorCollector,
    validate_synchronize_target,
)
from ._identity import validate_observation_name
from .actions import NonContiguousPolicy
from .errors import TensorCheckError, TensorDebugError, TensorOwnershipError
from .recording import (
    TensorObservation,
    _summarize_tensor,
)
from .snapshots import TensorCheckStatus, TensorProbeSnapshot

if TYPE_CHECKING:
    from .comparison import TensorComparisonOptions, TensorSnapshotComparison


class TensorProbe:
    """Transparent tensor probe that injects CUDA Graph debug side effects."""

    def __init__(
        self,
        name: str,
        actions: Sequence[TensorAction],
        *,
        non_contiguous: NonContiguousPolicy = "error",
        when: ProbeWhen = "capture",
        device: DeviceLike | None = None,
        synchronize: SynchronizeTarget = True,
    ) -> None:
        validate_synchronize_target(synchronize)
        self.synchronize = synchronize
        self.name = name
        self._probe_id = uuid.uuid4().hex
        self._closed = False
        self._collector = _TensorCollector(
            name,
            actions,
            non_contiguous=non_contiguous,
            when=when,
            device=device,
        )
        self.non_contiguous = self._collector.non_contiguous
        self.when = self._collector.when
        self._capture_invocation_counts: dict[str, int] = {}
        self._next_snapshot_index = 0

    @property
    def replay_index(self) -> torch.Tensor | None:
        """Return a detached GPU copy of the graph replay counter."""

        return self._collector.replay_index

    def __call__(
        self,
        tensor: torch.Tensor,
        *,
        name: str | None = None,
    ) -> torch.Tensor:
        """Return ``tensor`` unchanged while enqueueing one named observation."""

        resolved_name = validate_observation_name(self.name if name is None else name)
        invocation_index, counted = self._classify_invocation(
            tensor,
            resolved_name,
        )
        result = self._collector.enqueue(
            tensor,
            name=resolved_name,
            invocation_index=invocation_index,
        )
        # Commit the invocation index only after the enqueue succeeded: a
        # rejected call must not burn the index its retry will need.
        if counted:
            self._capture_invocation_counts[resolved_name] = invocation_index + 1
        return result

    def watch_grad(
        self,
        tensor: torch.Tensor,
        *,
        name: str | None = None,
        strict: bool = False,
    ) -> RemovableHandle | None:
        """Register an autograd hook that probes the tensor's backward gradient."""

        self._ensure_open()
        resolved_name = validate_observation_name(self.name if name is None else name)
        if not tensor.requires_grad:
            if strict:
                raise RuntimeError(
                    "cannot watch gradients for a tensor that does not require grad"
                )
            return None

        def hook(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            self(grad, name=resolved_name)
            return grad

        return tensor.register_hook(hook)

    def snapshot(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> TensorProbeSnapshot:
        """Return all recorded invocation values from the latest query point."""

        self._ensure_open()
        if not self._collector.record_enabled:
            raise TensorDebugError(
                "TensorProbe.snapshot() requires an enabled RecordAction"
            )
        selected = self.synchronize if synchronize is None else synchronize
        collected = self._collector.collect(synchronize=selected)
        if not collected:
            raise TensorDebugError("TensorProbe has no recorded invocation snapshot")
        replay_indices = {item.replay_index for item in collected}
        if len(replay_indices) != 1:
            raise TensorDebugError(
                "recorded tensor invocations reported different replay indices"
            )
        observations = []
        for item in collected:
            tensor = item.tensor.detach().contiguous().cpu()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            observations.append(
                TensorObservation(
                    order=item.order,
                    name=item.name,
                    invocation_index=item.invocation_index,
                    shape=item.shape,
                    stride=item.stride,
                    dtype=item.dtype,
                    source_device=item.source_device,
                    nbytes=len(raw),
                    payload="full",
                    sha256=hashlib.sha256(raw).hexdigest(),
                    summary=_summarize_tensor(tensor),
                    _tensor_cache={"tensor": tensor},
                    _cache_tensors=True,
                )
            )
        snapshot = TensorProbeSnapshot(
            probe_id=self._probe_id,
            probe_name=self.name,
            snapshot_index=self._next_snapshot_index,
            replay_index=next(iter(replay_indices)),
            timestamp=time.time(),
            observations=tuple(observations),
        )
        self._next_snapshot_index += 1
        return snapshot

    def compare(
        self,
        reference: TensorProbeSnapshot,
        candidate: TensorProbeSnapshot,
        *,
        options: TensorComparisonOptions | None = None,
    ) -> TensorSnapshotComparison:
        """Compare two chronologically ordered snapshots owned by this probe."""

        for role, snapshot in (
            ("reference", reference),
            ("candidate", candidate),
        ):
            if snapshot.probe_id != self._probe_id:
                raise TensorOwnershipError(
                    f"{role} snapshot does not belong to TensorProbe({self.name!r})"
                )
        if candidate.snapshot_index <= reference.snapshot_index:
            raise ValueError("candidate snapshot must follow reference snapshot")

        from .comparison import compare_snapshots

        return compare_snapshots(reference, candidate, options=options)

    def check_status(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> TensorCheckStatus:
        """Return the native CheckAction status after requested synchronization."""

        self._ensure_open()
        selected = self.synchronize if synchronize is None else synchronize
        native_check_status = self._collector.check_status(synchronize=selected)
        required = {
            "ok",
            "message",
            "replay_index",
            "order",
            "name",
            "invocation_index",
        }
        missing = required - set(native_check_status)
        if missing:
            raise TensorDebugError(
                f"native check status is missing fields {sorted(missing)!r}"
            )
        ok = native_check_status["ok"]
        message = native_check_status["message"]
        replay_index = native_check_status["replay_index"]
        order = native_check_status["order"]
        raw_name = native_check_status["name"]
        invocation_index = native_check_status["invocation_index"]
        if type(ok) is not bool:
            raise TensorDebugError("native check status ok must be a boolean")
        if not isinstance(message, str):
            raise TensorDebugError("native check status message must be a string")
        if type(replay_index) is not int or replay_index < 0:
            raise TensorDebugError(
                "native check status replay_index must be non-negative"
            )
        if type(order) is not int:
            raise TensorDebugError("native check status order must be an integer")
        if raw_name is not None and not isinstance(raw_name, str):
            raise TensorDebugError("native check status name must be a string or None")
        if type(invocation_index) is not int:
            raise TensorDebugError(
                "native check status invocation_index must be an integer"
            )
        return TensorCheckStatus(
            ok=ok,
            message=message,
            replay_index=replay_index,
            order=order,
            name=raw_name,
            invocation_index=invocation_index,
        )

    def assert_check_ok(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> None:
        """Synchronize as configured and raise for a callback mismatch."""

        check_status = self.check_status(synchronize=synchronize)
        if not check_status.ok:
            raise TensorCheckError(check_status.message or "tensor check failed")

    def close(
        self,
        *,
        synchronize: SynchronizeTarget | None = None,
    ) -> None:
        """Release native resources after captured graphs can no longer replay."""

        if not self._closed:
            selected = self.synchronize if synchronize is None else synchronize
            self._collector.close(synchronize=selected)
            self._closed = True

    def _classify_invocation(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> tuple[int, bool]:
        """Return ``(invocation_index, counted)`` without consuming the index."""

        if (
            not self._collector.enabled
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
        ):
            return 0, False
        with torch.cuda.device(tensor.device):
            if not torch.cuda.is_current_stream_capturing():
                return 0, False
        return self._capture_invocation_counts.get(name, 0), True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"TensorProbe({self.name!r}) is closed")

    def __enter__(self) -> "TensorProbe":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
