"""Object API for CUDA Graph tensor probes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, overload

import torch
from torch.utils.hooks import RemovableHandle

from torch_cudagraph_debug import _native

from .actions import (
    NonContiguousPolicy,
    TensorCompare,
    TensorPrint,
    TensorRecord,
    validate_non_contiguous_policy,
)
from .errors import TensorCompareMismatchError
from .records import TensorSnapshot

TensorAction = TensorPrint | TensorRecord | TensorCompare
ProbeMode = Literal["capture", "always"]


def validate_probe_mode(mode: str) -> ProbeMode:
    """Validate when a probe should enqueue debug work."""

    if mode not in {"capture", "always"}:
        raise ValueError('mode must be either "capture" or "always"')
    return mode  # type: ignore[return-value]


class CudaGraphTensorProbe:
    """Transparent tensor probe that injects CUDA Graph debug side effects."""

    def __init__(
        self,
        name: str,
        actions: Sequence[TensorAction],
        *,
        non_contiguous: NonContiguousPolicy = "error",
        mode: ProbeMode = "capture",
    ):
        if not name:
            raise ValueError("name must be non-empty")
        if not actions:
            raise ValueError("actions must be non-empty")

        self.name = name
        self._closed = False
        self._actions = tuple(actions)
        self.non_contiguous = validate_non_contiguous_policy(non_contiguous)
        self.mode = validate_probe_mode(mode)
        self._enabled_actions = tuple(
            action for action in self._actions if bool(action.enabled)
        )
        if self._enabled_actions:
            action_specs = [action._to_native() for action in self._enabled_actions]
            native = _native.require_native()
            self._handle: Any | None = native.create_tensor_debug_probe(
                name, action_specs, self.non_contiguous, self.mode
            )
        else:
            self._handle = None

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return ``tensor`` unchanged while enqueueing debug work on the current stream."""

        self._ensure_open()
        if self._handle is None:
            return tensor
        return self._handle.enqueue(tensor)

    @overload
    def attach_grad(
        self,
        tensor: torch.Tensor,
        *,
        strict: bool = False,
        return_handle: Literal[False] = False,
    ) -> torch.Tensor: ...

    @overload
    def attach_grad(
        self,
        tensor: torch.Tensor,
        *,
        strict: bool = False,
        return_handle: Literal[True],
    ) -> tuple[torch.Tensor, RemovableHandle | None]: ...

    def attach_grad(
        self,
        tensor: torch.Tensor,
        *,
        strict: bool = False,
        return_handle: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RemovableHandle | None]:
        """Register an autograd hook that probes ``tensor``'s backward gradient."""

        self._ensure_open()
        if not tensor.requires_grad:
            if strict:
                raise RuntimeError(
                    "cannot attach grad probe to a tensor that does not require grad"
                )
            if return_handle:
                return tensor, None
            return tensor

        def hook(grad: torch.Tensor | None) -> torch.Tensor | None:
            if grad is None:
                return None
            self(grad)
            return grad

        handle = tensor.register_hook(hook)
        if return_handle:
            return tensor, handle
        return tensor

    def records(self) -> list[TensorSnapshot]:
        """Return latest CPU tensor snapshots ordered by logical slot."""

        self._ensure_open()
        if self._handle is None:
            return []
        records: list[TensorSnapshot] = []
        for item in self._handle.records():
            tensor = item["tensor"]
            records.append(
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
        return records

    def clear_records(self) -> None:
        """Clear latest record buffers by zeroing their retained host storage."""

        self._ensure_open()
        if self._handle is None:
            return
        self._handle.clear_records()

    def status(self) -> dict[str, Any]:
        """Return native comparison status."""

        self._ensure_open()
        if self._handle is None:
            return {"ok": True, "message": "", "replay_index": 0, "invocation_index": -1}
        return dict(self._handle.status())

    def assert_ok(self) -> None:
        """Raise if any comparison action has reported a mismatch."""

        status = self.status()
        if not bool(status.get("ok", False)):
            message = str(status.get("message", "tensor comparison failed"))
            raise TensorCompareMismatchError(message)

    def close(self) -> None:
        """Release native resources.

        The caller must guarantee that no CUDA graph which captured this probe can replay again.
        """

        if not self._closed:
            if self._handle is not None:
                self._handle.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"CudaGraphTensorProbe({self.name!r}) is closed")

    def __enter__(self) -> "CudaGraphTensorProbe":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
