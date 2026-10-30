from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from uvicorn.config import Config

#: How lifespan failures are handled for a run segment.
#:
#: * ``pin`` - keep the existing behaviour: a failed startup aborts process
#:   startup (with the usual shutdown ordering), and connection faults do not
#:   trigger a shutdown on their own.
#: * ``suppress_startup`` - a failed ``lifespan.startup`` is logged but the
#:   server still enters the main loop. ``Config.lifespan`` is never rewritten.
#: * ``fail_fast_shutdown`` - an unrecoverable exception while handling a
#:   connection triggers ``Server.shutdown()`` immediately, while the lifespan
#:   shutdown callback is still awaited.
LifespanFault: TypeAlias = Literal["pin", "suppress_startup", "fail_fast_shutdown"]

#: Whether the HTTP access line is emitted for a request frame.
#:
#: * ``on`` - emit the access line (subject to the configured access logger).
#: * ``off`` - never write the access logger; the application/error loggers are
#:   untouched.
#: * ``errors_only`` - only emit for ``5xx`` responses or protocol errors.
AccessEmit: TypeAlias = Literal["on", "off", "errors_only"]

#: How a connection is torn down at the end of a request frame.
#:
#: * ``pin`` - keep the existing keep-alive / half-close behaviour.
#: * ``drain_then_close`` - once the response is fully written, drain the
#:   unread request body before closing the connection.
#: * ``force_close`` - close as soon as the response for the frame is
#:   complete, ignoring keep-alive. An application response iterator that has
#:   not finished yet is never torn down early.
ConnHalfClose: TypeAlias = Literal["pin", "drain_then_close", "force_close"]

_LIFESPAN_FAULTS = frozenset({"pin", "suppress_startup", "fail_fast_shutdown"})
_ACCESS_EMITS = frozenset({"on", "off", "errors_only"})
_CONN_HALF_CLOSES = frozenset({"pin", "drain_then_close", "force_close"})


@dataclass(frozen=True, slots=True)
class PolicyLayer:
    """A partial override.

    Every dimension defaults to ``None``, meaning "do not decide here; fall
    through to the layer below (and finally to the :class:`~uvicorn.config.Config`
    baseline)". Layers therefore compose: an inner layer only needs to mention
    the dimensions it overrides.
    """

    lifespan_fault: LifespanFault | None = None
    access_emit: AccessEmit | None = None
    conn_half_close: ConnHalfClose | None = None

    def __post_init__(self) -> None:
        if self.lifespan_fault is not None and self.lifespan_fault not in _LIFESPAN_FAULTS:
            raise ValueError(f"Invalid lifespan_fault: {self.lifespan_fault!r}")
        if self.access_emit is not None and self.access_emit not in _ACCESS_EMITS:
            raise ValueError(f"Invalid access_emit: {self.access_emit!r}")
        if self.conn_half_close is not None and self.conn_half_close not in _CONN_HALF_CLOSES:
            raise ValueError(f"Invalid conn_half_close: {self.conn_half_close!r}")


@dataclass(frozen=True, slots=True)
class EffectivePolicy:
    """The fully resolved view for a frame: every dimension is concrete.

    This is the result of merging every active layer with the
    :class:`~uvicorn.config.Config` baseline, so callers never inspect
    ``Config`` flat fields themselves.
    """

    lifespan_fault: LifespanFault
    access_emit: AccessEmit
    conn_half_close: ConnHalfClose


#: The baseline used before anything is pushed (and when the stack is empty).
#: ``pin``/``on``/``pin`` reproduce the current on-the-wire behaviour; the
#: existing access-logger handler gate still honours ``Config.access_log``.
DEFAULT_BASELINE = EffectivePolicy(lifespan_fault="pin", access_emit="on", conn_half_close="pin")


class _Node:
    __slots__ = ("layer", "prev")

    def __init__(self, layer: PolicyLayer, prev: _Node | None) -> None:
        self.layer = layer
        self.prev = prev


class _PopToken:
    __slots__ = ("stack", "node")

    def __init__(self, stack: PolicyStack, node: _Node) -> None:
        self.stack = stack
        self.node = node


# Frame-local layers. Context variables are copied per task and per
# ``Context.run`` boundary, so two concurrent connections never share a frame
# layer pushed inside their own handling task.
_frame_head: contextvars.ContextVar[_Node | None] = contextvars.ContextVar("uvicorn_policy_frame_head", default=None)


def _resolve(server_head: _Node | None, baseline: EffectivePolicy) -> EffectivePolicy:
    """Resolve dimensions with priority: frame-local > server stack > baseline.

    Within each chain the nearest layer wins. Both chains are immutable, so the
    ``server_head`` captured by a snapshot is unaffected by later pushes/pops.
    """

    lifespan_fault: LifespanFault | None = None
    access_emit: AccessEmit | None = None
    conn_half_close: ConnHalfClose | None = None

    node = _frame_head.get()
    while node is not None:
        layer = node.layer
        if lifespan_fault is None and layer.lifespan_fault is not None:
            lifespan_fault = layer.lifespan_fault
        if access_emit is None and layer.access_emit is not None:
            access_emit = layer.access_emit
        if conn_half_close is None and layer.conn_half_close is not None:
            conn_half_close = layer.conn_half_close
        node = node.prev

    node = server_head
    while node is not None:
        layer = node.layer
        if lifespan_fault is None and layer.lifespan_fault is not None:
            lifespan_fault = layer.lifespan_fault
        if access_emit is None and layer.access_emit is not None:
            access_emit = layer.access_emit
        if conn_half_close is None and layer.conn_half_close is not None:
            conn_half_close = layer.conn_half_close
        node = node.prev

    return EffectivePolicy(
        lifespan_fault=lifespan_fault or baseline.lifespan_fault,
        access_emit=access_emit or baseline.access_emit,
        conn_half_close=conn_half_close or baseline.conn_half_close,
    )


def access_enabled(effective: EffectivePolicy, status_code: int, has_handler: bool) -> bool:
    """Decide whether the access line for ``status_code`` must be written.

    ``has_handler`` is the pre-existing ``access_logger.hasHandlers()`` gate,
    so ``Config.access_log=False`` keeps suppressing every access line.
    Application and error loggers are deliberately not consulted here.
    """

    if not has_handler:
        return False
    mode = effective.access_emit
    if mode == "off":
        return False
    if mode == "errors_only":
        return status_code >= 500
    return True


def protocol_error_access_enabled(effective: EffectivePolicy, has_handler: bool) -> bool:
    """Whether a malformed-request (protocol error) access line is emitted."""

    return has_handler and effective.access_emit == "errors_only"


@contextlib.contextmanager
def frame_scope(
    *,
    lifespan_fault: LifespanFault | None = None,
    access_emit: AccessEmit | None = None,
    conn_half_close: ConnHalfClose | None = None,
) -> Iterator[None]:
    """Push a layer bound to the current request/connection frame only.

    The layer rides a :class:`contextvars.ContextVar`, so it is isolated to the
    current task (and any task created while it is active). It is popped
    automatically, including on the exception path.
    """

    layer = PolicyLayer(lifespan_fault=lifespan_fault, access_emit=access_emit, conn_half_close=conn_half_close)
    token = _frame_head.set(_Node(layer, _frame_head.get()))
    try:
        yield
    finally:
        _frame_head.reset(token)


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """An immutable point-in-time capture of the server stack for one frame.

    A request cycle takes a snapshot when it starts and resolves every later
    decision against it, so pushes/pops made by another in-flight connection
    can never leak into this frame ("no cross-frame interference").
    """

    _stack: PolicyStack
    _head: _Node | None

    def effective(self) -> EffectivePolicy:
        return _resolve(self._head, self._stack.baseline)

    def notify_uncaught(self) -> None:
        """Report an unrecoverable connection-handling fault.

        Only acts when the effective ``lifespan_fault`` is
        ``fail_fast_shutdown``; otherwise it is a no-op.
        """

        if self.effective().lifespan_fault != "fail_fast_shutdown":
            return
        request_shutdown = self._stack.request_shutdown
        if request_shutdown is not None:
            request_shutdown()


class PolicyStack:
    """A per-Server LIFO stack of run-policy layers.

    The stack is owned by :class:`~uvicorn.server.ServerState`; it is never a
    process-wide singleton, so multiple workers, servers and concurrent
    TestClient-style drivers each get an independent stack. An empty stack
    resolves to the :class:`~uvicorn.config.Config` baseline, i.e. the current
    production behaviour.
    """

    def __init__(self, baseline: EffectivePolicy = DEFAULT_BASELINE) -> None:
        self._head: _Node | None = None
        self._baseline = baseline
        #: Wired by the Server; requests a (lifespan-preserving) shutdown.
        self.request_shutdown: Callable[[], None] | None = None

    @property
    def baseline(self) -> EffectivePolicy:
        return self._baseline

    def bind_config(self, config: Config) -> None:
        """Derive the static baseline dimensions from a loaded Config."""
        access_emit: AccessEmit = "on" if config.access_log else "off"
        self._baseline = EffectivePolicy("pin", access_emit, "pin")

    def __bool__(self) -> bool:
        return self._head is not None

    def push(
        self,
        *,
        lifespan_fault: LifespanFault | None = None,
        access_emit: AccessEmit | None = None,
        conn_half_close: ConnHalfClose | None = None,
    ) -> _PopToken:
        layer = PolicyLayer(lifespan_fault=lifespan_fault, access_emit=access_emit, conn_half_close=conn_half_close)
        node = _Node(layer, self._head)
        self._head = node
        return _PopToken(self, node)

    def pop(self, token: _PopToken) -> None:
        """Pop the layer identified by ``token``.

        Popping anything other than the current top is an error: an inner
        segment must be removed before the outer one, which also makes a
        mismatched pop unable to empty a stack it does not own.
        """

        if token.stack is not self:
            raise RuntimeError("Cannot pop a policy layer on a different PolicyStack.")
        if self._head is None or self._head is not token.node:
            raise RuntimeError("Cannot pop a policy layer that is not the current stack top.")
        self._head = token.node.prev

    @contextlib.contextmanager
    def scope(
        self,
        *,
        lifespan_fault: LifespanFault | None = None,
        access_emit: AccessEmit | None = None,
        conn_half_close: ConnHalfClose | None = None,
    ) -> Iterator[PolicySnapshot]:
        token = self.push(lifespan_fault=lifespan_fault, access_emit=access_emit, conn_half_close=conn_half_close)
        try:
            yield self.snapshot()
        finally:
            self.pop(token)

    def snapshot(self) -> PolicySnapshot:
        return PolicySnapshot(self, self._head)

    def effective(self, snapshot: PolicySnapshot | None = None) -> EffectivePolicy:
        """Effective view.

        With ``snapshot`` (or the live stack when omitted) merged with frame
        local layers and the Config baseline.
        """

        head = snapshot._head if snapshot is not None else self._head
        return _resolve(head, self._baseline)

    def current(self) -> EffectivePolicy:
        """Live effective view (for lifespan steps that run in the main task)."""

        return _resolve(self._head, self._baseline)
