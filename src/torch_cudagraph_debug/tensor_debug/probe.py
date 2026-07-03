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
)
from .actions import NonContiguousPolicy
from .errors import TensorCheckError, TensorDebugError, TensorOwnershipError
from .recording import (
    TensorObservation,
    _summarize_tensor,
    _validate_observation_name,
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
    ):
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

        resolved_name = _validate_observation_name(self.name if name is None else name)
        invocation_index = self._next_capture_invocation(
            tensor,
            resolved_name,
        )
        return self._collector.enqueue(
            tensor,
            name=resolved_name,
            invocation_index=invocation_index,
        )

    def watch_grad(
        self,
        tensor: torch.Tensor,
        *,
        name: str | None = None,
        strict: bool = False,
    ) -> RemovableHandle | None:
        """Register an autograd hook that probes the tensor's backward gradient."""

        self._ensure_open()
        resolved_name = _validate_observation_name(self.name if name is None else name)
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
        synchronize: SynchronizeTarget = True,
    ) -> TensorProbeSnapshot:
        """Return all recorded invocation values from the latest replay.

        ``True`` synchronizes the probe device, a CUDA stream synchronizes only
        that stream, a CUDA device explicitly synchronizes that device, and
        ``False`` leaves ordering to the caller.
        """

        self._ensure_open()
        if not self._collector.record_enabled:
            raise TensorDebugError(
                "TensorProbe.snapshot() requires an enabled RecordAction"
            )
        collected = self._collector.collect(synchronize=synchronize)
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
                    stride=tuple(tensor.stride()),
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
        return TensorProbeSnapshot(
            probe_id=self._probe_id,
            probe_name=self.name,
            replay_index=next(iter(replay_indices)),
            timestamp=time.time(),
            observations=tuple(observations),
        )

    def compare(
        self,
        reference: TensorProbeSnapshot,
        candidate: TensorProbeSnapshot,
        *,
        options: TensorComparisonOptions | None = None,
    ) -> TensorSnapshotComparison:
        """Compare two chronologically ordered snapshots owned by this probe."""

        self._ensure_open()
        for role, snapshot in (
            ("reference", reference),
            ("candidate", candidate),
        ):
            if snapshot.probe_id != self._probe_id:
                raise TensorOwnershipError(
                    f"{role} snapshot does not belong to TensorProbe({self.name!r})"
                )
        if candidate.replay_index <= reference.replay_index:
            raise ValueError("candidate snapshot must follow reference snapshot")

        from .comparison import compare_snapshots

        return compare_snapshots(reference, candidate, options=options)

    def clear_snapshot(
        self,
        *,
        synchronize: SynchronizeTarget = True,
    ) -> None:
        """Synchronize as requested, then clear retained host snapshot storage."""

        self._ensure_open()
        self._collector.clear(synchronize=synchronize)

    def check_status(
        self,
        *,
        synchronize: SynchronizeTarget = True,
    ) -> TensorCheckStatus:
        """Return the native CheckAction status after requested synchronization."""

        self._ensure_open()
        native_check_status = self._collector.check_status(synchronize=synchronize)
        raw_name = native_check_status.get("name")
        return TensorCheckStatus(
            ok=bool(native_check_status.get("ok", False)),
            message=str(native_check_status.get("message", "")),
            replay_index=int(native_check_status.get("replay_index", 0)),
            order=int(native_check_status.get("order", -1)),
            name=None if raw_name is None else str(raw_name),
            invocation_index=int(native_check_status.get("invocation_index", -1)),
        )

    def assert_check_ok(
        self,
        *,
        synchronize: SynchronizeTarget = True,
    ) -> None:
        """Synchronize once and raise if a callback reported a mismatch."""

        check_status = self.check_status(synchronize=synchronize)
        if not check_status.ok:
            raise TensorCheckError(check_status.message or "tensor check failed")

    def close(
        self,
        *,
        synchronize: SynchronizeTarget = True,
    ) -> None:
        """Synchronize as requested, then release native resources.

        The caller must guarantee that no CUDA graph which captured this probe
        can replay again.
        """

        if not self._closed:
            self._collector.close(synchronize=synchronize)
            self._closed = True

    def _next_capture_invocation(
        self,
        tensor: torch.Tensor,
        name: str,
    ) -> int:
        if (
            not self._collector.enabled
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cuda"
        ):
            return 0
        with torch.cuda.device(tensor.device):
            if not torch.cuda.is_current_stream_capturing():
                return 0
        invocation_index = self._capture_invocation_counts.get(name, 0)
        self._capture_invocation_counts[name] = invocation_index + 1
        return invocation_index

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"TensorProbe({self.name!r}) is closed")

    def __enter__(self) -> "TensorProbe":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
