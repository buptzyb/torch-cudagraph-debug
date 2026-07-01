"""Object API for CUDA Graph tensor probes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch.utils.hooks import RemovableHandle

from torch_cudagraph_debug import _native

from .actions import (
    NonContiguousPolicy,
    CompareTensor,
    PrintTensor,
    RecordTensor,
    validate_non_contiguous_policy,
)
from .errors import TensorMismatchError
from .records import TensorProbeStatus, TensorSnapshot

TensorAction = PrintTensor | RecordTensor | CompareTensor
ProbeWhen = Literal["capture", "always"]
DeviceLike = torch.device | str | int


def _create_replay_index(device: DeviceLike | None) -> torch.Tensor:
    if device is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    elif isinstance(device, int) and not isinstance(device, bool):
        resolved = torch.device("cuda", device)
    else:
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("device must identify a CUDA device")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())

    return torch.zeros((), dtype=torch.int64, device=resolved)


def _validate_synchronize_target(
    synchronize: bool | torch.cuda.Stream | torch.device,
) -> None:
    if not isinstance(synchronize, (bool, torch.cuda.Stream, torch.device)):
        raise TypeError(
            "synchronize must be a bool, torch.cuda.Stream, or torch.device"
        )


def _indexed_cuda_device(device: torch.device, *, argument: str) -> torch.device:
    if device.type != "cuda":
        raise ValueError(f"{argument} must identify a CUDA device")
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _synchronize_probe_results(
    probe_device: torch.device,
    synchronize: bool | torch.cuda.Stream | torch.device,
) -> None:
    if isinstance(synchronize, bool):
        if not synchronize:
            return
        target_device = probe_device
        target_stream = None
    elif isinstance(synchronize, torch.cuda.Stream):
        target_device = _indexed_cuda_device(
            synchronize.device,
            argument="synchronize stream device",
        )
        target_stream = synchronize
    else:
        target_device = _indexed_cuda_device(
            synchronize,
            argument="synchronize device",
        )
        target_stream = None

    if target_device != probe_device:
        raise ValueError(
            f"synchronize target {target_device} does not match probe device "
            f"{probe_device}"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "cannot synchronize TensorProbe results during CUDA graph capture; "
            "query after capture or pass synchronize=False"
        )

    if target_stream is not None:
        target_stream.synchronize()
    else:
        torch.cuda.synchronize(target_device)


def validate_probe_when(when: str) -> ProbeWhen:
    """Validate when a probe should enqueue debug work."""

    if when not in {"capture", "always"}:
        raise ValueError('when must be either "capture" or "always"')
    return when  # type: ignore[return-value]


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
        if not name:
            raise ValueError("name must be non-empty")
        if not actions:
            raise ValueError("actions must be non-empty")

        self.name = name
        self._closed = False
        self._actions = tuple(actions)
        self.non_contiguous = validate_non_contiguous_policy(non_contiguous)
        self.when = validate_probe_when(when)
        self._enabled_actions = tuple(
            action for action in self._actions if bool(action.enabled)
        )
        self._records_enabled = any(
            isinstance(action, RecordTensor) for action in self._enabled_actions
        )
        self._callback_actions_enabled = any(
            isinstance(action, (PrintTensor, CompareTensor))
            for action in self._enabled_actions
        )
        self._replay_index: torch.Tensor | None = None
        self._device: torch.device | None = None
        if self._enabled_actions:
            native = _native.require_native()
            self._replay_index = _create_replay_index(device)
            self._device = self._replay_index.device
            action_specs = [action._to_native() for action in self._enabled_actions]
            self._handle: Any | None = native.create_tensor_debug_probe(
                name,
                action_specs,
                self._replay_index,
                self.non_contiguous,
                self.when,
            )
        else:
            self._handle = None

    @property
    def replay_index(self) -> torch.Tensor | None:
        """Return a detached GPU copy of the graph replay counter."""

        self._ensure_open()
        if self._replay_index is None:
            return None
        return self._replay_index.detach().clone()

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` unchanged while enqueueing debug work on the current stream."""

        self._ensure_open()
        if self._handle is None:
            return tensor
        return self._handle.enqueue(tensor)

    def watch_grad(
        self,
        tensor: torch.Tensor,
        *,
        strict: bool = False,
    ) -> RemovableHandle | None:
        """Register an autograd hook that probes the tensor's backward gradient."""

        self._ensure_open()
        if not tensor.requires_grad:
            if strict:
                raise RuntimeError(
                    "cannot watch gradients for a tensor that does not require grad"
                )
            return None

        def hook(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            self(grad)
            return grad

        return tensor.register_hook(hook)

    def snapshots(
        self,
        *,
        synchronize: bool | torch.cuda.Stream | torch.device = True,
    ) -> list[TensorSnapshot]:
        """Return latest CPU snapshots after applying the requested synchronization.

        ``True`` synchronizes the probe device, a CUDA stream synchronizes only
        that stream, a CUDA device explicitly synchronizes that device, and
        ``False`` leaves ordering to the caller.
        """

        self._ensure_open()
        _validate_synchronize_target(synchronize)
        if self._handle is None or not self._records_enabled:
            return []
        if self._replay_index is None:
            raise RuntimeError("enabled tensor probe is missing its replay counter")
        self._synchronize_results(synchronize)
        current_replay_index = (
            None
            if self._callback_actions_enabled
            else int(self._replay_index.item())
        )
        snapshots: list[TensorSnapshot] = []
        for item in self._handle.records(current_replay_index):
            tensor = item["tensor"]
            snapshots.append(
                TensorSnapshot(
                    probe_name=str(item["probe_name"]),
                    replay_index=int(item["replay_index"]),
                    tensor=tensor,
                    shape=tuple(item.get("shape", tuple(tensor.shape))),
                    dtype=getattr(tensor, "dtype"),
                    device=str(item.get("device", "")),
                    invocation_index=int(item.get("invocation_index", 0)),
                )
            )
        return snapshots

    def clear_snapshots(
        self,
        *,
        synchronize: bool | torch.cuda.Stream | torch.device = True,
    ) -> None:
        """Synchronize as requested, then zero retained host record storage."""

        self._ensure_open()
        _validate_synchronize_target(synchronize)
        if self._handle is None or not self._records_enabled:
            return
        self._synchronize_results(synchronize)
        self._handle.clear_records()

    def status(
        self,
        *,
        synchronize: bool | torch.cuda.Stream | torch.device = True,
    ) -> TensorProbeStatus:
        """Return native callback status after requested synchronization."""

        self._ensure_open()
        _validate_synchronize_target(synchronize)
        if self._handle is None:
            return TensorProbeStatus(True, "", 0, -1)
        if self._callback_actions_enabled:
            self._synchronize_results(synchronize)
        status = self._handle.status()
        return TensorProbeStatus(
            ok=bool(status.get("ok", False)),
            message=str(status.get("message", "")),
            replay_index=int(status.get("replay_index", 0)),
            invocation_index=int(status.get("invocation_index", -1)),
        )

    def assert_ok(
        self,
        *,
        synchronize: bool | torch.cuda.Stream | torch.device = True,
    ) -> None:
        """Synchronize once and raise if a callback reported a mismatch."""

        status = self.status(synchronize=synchronize)
        if not status.ok:
            raise TensorMismatchError(status.message or "tensor comparison failed")

    def close(self) -> None:
        """Release native resources.

        The caller must guarantee that no CUDA graph which captured this probe can replay again.
        """

        if not self._closed:
            if self._handle is not None:
                self._handle.close()
            self._handle = None
            self._replay_index = None
            self._device = None
            self._closed = True

    def _synchronize_results(
        self,
        synchronize: bool | torch.cuda.Stream | torch.device,
    ) -> None:
        if self._device is None:
            raise RuntimeError("enabled tensor probe is missing its CUDA device")
        _synchronize_probe_results(self._device, synchronize)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"TensorProbe({self.name!r}) is closed")

    def __enter__(self) -> "TensorProbe":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
