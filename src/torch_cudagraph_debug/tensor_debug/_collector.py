"""Private tensor acquisition used by Probe and Recorder workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from torch_cudagraph_debug import _native

from .actions import (
    CheckAction,
    NonContiguousPolicy,
    PrintAction,
    RecordAction,
    validate_non_contiguous_policy,
)

TensorAction = PrintAction | RecordAction | CheckAction
ProbeWhen = Literal["capture", "always"]
DeviceLike = torch.device | str | int
SynchronizeTarget = bool | torch.cuda.Stream | torch.device


@dataclass(frozen=True)
class _CollectedTensor:
    probe_name: str
    replay_index: int
    invocation_index: int
    tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    source_device: str


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


def validate_synchronize_target(synchronize: SynchronizeTarget) -> None:
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


def synchronize_tensor_results(
    collector_device: torch.device,
    synchronize: SynchronizeTarget,
) -> None:
    validate_synchronize_target(synchronize)
    if isinstance(synchronize, bool):
        if not synchronize:
            return
        target_device = collector_device
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

    if target_device != collector_device:
        raise ValueError(
            f"synchronize target {target_device} does not match collector device "
            f"{collector_device}"
        )
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "cannot synchronize tensor collector results during CUDA graph capture; "
            "query after capture or pass synchronize=False"
        )

    if target_stream is not None:
        target_stream.synchronize()
    else:
        torch.cuda.synchronize(target_device)


def validate_probe_when(when: str) -> ProbeWhen:
    if when not in {"capture", "always"}:
        raise ValueError('when must be either "capture" or "always"')
    return when  # type: ignore[return-value]


class _EagerTensorCollector:
    """Enqueue eager device-to-host copies and synchronize their completion."""

    @staticmethod
    def collect(source: torch.Tensor) -> torch.Tensor:
        staging = torch.empty(
            tuple(source.shape),
            dtype=source.dtype,
            device="cpu",
            pin_memory=True,
        )
        staging.copy_(source.detach(), non_blocking=True)
        return staging

    @staticmethod
    def synchronize(
        device: torch.device,
        synchronize: SynchronizeTarget,
    ) -> None:
        synchronize_tensor_results(device, synchronize)


class _TensorCollector:
    """Own the native probe handle and return raw collected tensors."""

    def __init__(
        self,
        name: str,
        actions: Sequence[TensorAction],
        *,
        non_contiguous: NonContiguousPolicy = "error",
        when: ProbeWhen = "capture",
        device: DeviceLike | None = None,
    ) -> None:
        if not name:
            raise ValueError("name must be non-empty")
        if not actions:
            raise ValueError("actions must be non-empty")

        self.name = name
        self.non_contiguous = validate_non_contiguous_policy(non_contiguous)
        self.when = validate_probe_when(when)
        self._closed = False
        self._actions = tuple(actions)
        self._enabled_actions = tuple(
            action for action in self._actions if bool(action.enabled)
        )
        self.record_enabled = any(
            isinstance(action, RecordAction) for action in self._enabled_actions
        )
        self.callback_enabled = any(
            isinstance(action, (PrintAction, CheckAction))
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
    def device(self) -> torch.device | None:
        return self._device

    @property
    def replay_index(self) -> torch.Tensor | None:
        self._ensure_open()
        if self._replay_index is None:
            return None
        return self._replay_index.detach().clone()

    def enqueue(self, tensor: torch.Tensor) -> torch.Tensor:
        self._ensure_open()
        if self._handle is None:
            return tensor
        return self._handle.enqueue(tensor)

    def collect(
        self,
        *,
        synchronize: SynchronizeTarget,
    ) -> tuple[_CollectedTensor, ...]:
        self._ensure_open()
        validate_synchronize_target(synchronize)
        if self._handle is None or not self.record_enabled:
            return ()
        if self._replay_index is None:
            raise RuntimeError("enabled tensor collector is missing its replay counter")
        self.synchronize(synchronize)
        current_replay_index = (
            None if self.callback_enabled else int(self._replay_index.item())
        )
        output = []
        for item in self._handle.observations(current_replay_index):
            tensor = item["tensor"]
            output.append(
                _CollectedTensor(
                    probe_name=str(item["probe_name"]),
                    replay_index=int(item["replay_index"]),
                    invocation_index=int(item.get("invocation_index", 0)),
                    tensor=tensor,
                    shape=tuple(item.get("shape", tuple(tensor.shape))),
                    dtype=getattr(tensor, "dtype"),
                    source_device=str(item.get("device", "")),
                )
            )
        return tuple(output)

    def clear(self, *, synchronize: SynchronizeTarget) -> None:
        self._ensure_open()
        validate_synchronize_target(synchronize)
        if self._handle is None or not self.record_enabled:
            return
        self.synchronize(synchronize)
        self._handle.clear_observations()

    def check_status(
        self,
        *,
        synchronize: SynchronizeTarget,
    ) -> Mapping[str, object]:
        self._ensure_open()
        validate_synchronize_target(synchronize)
        if self._handle is None:
            return {
                "ok": True,
                "message": "",
                "replay_index": 0,
                "invocation_index": -1,
            }
        if self.callback_enabled:
            self.synchronize(synchronize)
        return self._handle.check_status()

    def synchronize(self, synchronize: SynchronizeTarget) -> None:
        if self._device is None:
            raise RuntimeError("enabled tensor collector is missing its CUDA device")
        synchronize_tensor_results(self._device, synchronize)

    def close(self) -> None:
        if self._closed:
            return
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._replay_index = None
        self._device = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"tensor collector {self.name!r} is closed")
