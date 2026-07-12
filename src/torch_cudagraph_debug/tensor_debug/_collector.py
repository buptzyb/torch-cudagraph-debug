"""Private tensor acquisition used by Probe and Recorder workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from torch_cudagraph_debug import _native

from ._availability import require_live_tensor_debug
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
    name: str
    order: int
    replay_index: int
    invocation_index: int
    tensor: torch.Tensor
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    source_device: str
    eager_overwrites: int = 0


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
    if _is_capturing_on_device(target_device):
        raise RuntimeError(
            "cannot synchronize tensor collector results during CUDA Graph "
            "capture; query after capture or pass synchronize=False"
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
        self._source_strides: dict[tuple[str, int], tuple[int, ...]] = {}
        self._captured_once = False
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
            require_live_tensor_debug()
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
    def enabled(self) -> bool:
        return self._handle is not None

    @property
    def device(self) -> torch.device | None:
        return self._device

    @property
    def replay_index(self) -> torch.Tensor | None:
        self._ensure_open()
        if self._replay_index is None:
            return None
        return self._replay_index.detach().clone()

    def enqueue(
        self,
        tensor: torch.Tensor,
        *,
        name: str,
        invocation_index: int,
    ) -> torch.Tensor:
        self._ensure_open()
        if self._handle is None:
            return tensor
        assert self._device is not None
        capturing = _is_capturing_on_device(self._device)
        result = self._handle.enqueue(tensor, name, invocation_index)
        if self.when == "capture" and not capturing:
            return result
        if capturing and not self._captured_once:
            # Native capture replaces the eager name layout. Mirror that
            # transition so stale eager stride metadata does not survive.
            self._source_strides.clear()
            self._captured_once = True
        # Record stride bookkeeping only for enqueues the native layer
        # accepted; a rejected call must not leave a stale entry behind.
        self._source_strides[(name, invocation_index)] = tuple(tensor.stride())
        return result

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
        assert self._device is not None
        capturing = _is_capturing_on_device(self._device)
        # Freeing retired pinned staging is a CUDA call that would invalidate
        # an active capture; defer it to the next out-of-capture query.
        if not capturing:
            self._handle._reclaim_retired_staging()
        current_replay_index = None
        if not self.callback_enabled:
            if capturing:
                # Reading the replay counter is a blocking device read that
                # would invalidate an active capture.
                raise RuntimeError(
                    "cannot collect record-only tensor results during CUDA "
                    "graph capture; query after capture ends"
                )
            current_replay_index = int(self._replay_index.item())
        output = []
        for index, item in enumerate(self._handle.observations(current_replay_index)):
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"native tensor observation {index} must be a mapping"
                )
            required = {
                "name",
                "order",
                "replay_index",
                "invocation_index",
                "shape",
                "device",
                "tensor",
            }
            missing = required - set(item)
            if missing:
                raise RuntimeError(
                    f"native tensor observation {index} is missing fields "
                    f"{sorted(missing)!r}"
                )
            tensor = item["tensor"]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"native tensor observation {index} tensor must be a torch.Tensor"
                )
            name = item["name"]
            if not isinstance(name, str) or not name:
                raise RuntimeError(
                    f"native tensor observation {index} name must be non-empty"
                )
            order = _native_nonnegative_int(item["order"], f"observation {index} order")
            replay_index = _native_nonnegative_int(
                item["replay_index"], f"observation {index} replay_index"
            )
            invocation_index = _native_nonnegative_int(
                item["invocation_index"],
                f"observation {index} invocation_index",
            )
            eager_overwrites = _native_nonnegative_int(
                item.get("eager_overwrites", 0),
                f"observation {index} eager_overwrites",
            )
            raw_shape = item["shape"]
            if isinstance(raw_shape, (str, bytes)) or not isinstance(
                raw_shape, Sequence
            ):
                raise RuntimeError(
                    f"native tensor observation {index} shape must be a sequence"
                )
            shape = tuple(
                _native_nonnegative_int(value, f"observation {index} shape")
                for value in raw_shape
            )
            source_device = item["device"]
            if not isinstance(source_device, str) or not source_device:
                raise RuntimeError(
                    f"native tensor observation {index} device must be non-empty"
                )
            output.append(
                _CollectedTensor(
                    name=name,
                    order=order,
                    replay_index=replay_index,
                    invocation_index=invocation_index,
                    tensor=tensor,
                    shape=shape,
                    stride=self._source_strides[(name, invocation_index)],
                    dtype=tensor.dtype,
                    source_device=source_device,
                    eager_overwrites=eager_overwrites,
                )
            )
        return tuple(output)

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
                "order": -1,
                "name": None,
                "invocation_index": -1,
            }
        if self.callback_enabled:
            self.synchronize(synchronize)
            assert self._device is not None
            if not _is_capturing_on_device(self._device):
                self._handle._reclaim_retired_staging()
        return self._handle.check_status()

    def synchronize(self, synchronize: SynchronizeTarget) -> None:
        if self._device is None:
            raise RuntimeError("enabled tensor collector is missing its CUDA device")
        synchronize_tensor_results(self._device, synchronize)

    def close(
        self,
        *,
        synchronize: SynchronizeTarget = True,
    ) -> None:
        if self._closed:
            return
        validate_synchronize_target(synchronize)
        if self._handle is not None:
            assert self._device is not None
            if _is_capturing_on_device(self._device):
                raise RuntimeError(
                    "cannot close tensor collector during CUDA Graph capture"
                )
            self.synchronize(synchronize)
            self._handle.close()
        self._handle = None
        self._replay_index = None
        self._device = None
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"tensor collector {self.name!r} is closed")


def _native_nonnegative_int(value: object, context: str) -> int:
    if type(value) is not int or value < 0:
        raise RuntimeError(f"native tensor {context} must be a non-negative integer")
    return value


def _is_capturing_on_device(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return bool(torch.cuda.is_current_stream_capturing())
    with torch.cuda.device(device):
        return bool(torch.cuda.is_current_stream_capturing())
