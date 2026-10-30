from __future__ import annotations

import asyncio
import contextlib
import logging

import pytest

from uvicorn.config import Config
from uvicorn.lifespan.on import LifespanOn
from uvicorn.main import Server
from uvicorn.policy import (
    DEFAULT_BASELINE,
    EffectivePolicy,
    PolicyLayer,
    PolicyStack,
    frame_scope,
)
from uvicorn.server import ServerState


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# Policy stack semantics
# --------------------------------------------------------------------------- #


def test_empty_stack_matches_baseline():
    stack = PolicyStack()
    assert not stack
    assert stack.current() == EffectivePolicy("pin", "on", "pin")
    assert stack.current() == DEFAULT_BASELINE


def test_partial_layer_leaves_other_dimensions_at_baseline():
    stack = PolicyStack()
    with stack.scope(conn_half_close="force_close"):
        eff = stack.current()
        assert eff.conn_half_close == "force_close"
        assert eff.lifespan_fault == "pin"
        assert eff.access_emit == "on"
    assert not stack


def test_nested_layers_inner_wins_then_outer_is_restored():
    stack = PolicyStack()
    with stack.scope(access_emit="off"):
        assert stack.current().access_emit == "off"
        with stack.scope(access_emit="errors_only"):
            assert stack.current().access_emit == "errors_only"
        assert stack.current().access_emit == "off"
    assert stack.current().access_emit == "on"


def test_pop_of_non_top_is_rejected():
    stack = PolicyStack()
    outer = stack.push(access_emit="off")
    stack.push(access_emit="errors_only")
    with pytest.raises(RuntimeError):
        stack.pop(outer)


def test_scope_pops_even_on_exception():
    stack = PolicyStack()
    with pytest.raises(RuntimeError):
        with stack.scope(access_emit="off", lifespan_fault="suppress_startup"):
            raise RuntimeError("boom")
    assert not stack
    assert stack.current() == DEFAULT_BASELINE


def test_snapshot_is_frozen_against_later_push_and_pop():
    stack = PolicyStack()
    with stack.scope(access_emit="off"):
        snapshot = stack.snapshot()
        with stack.scope(access_emit="errors_only"):
            # The earlier frame keeps seeing "off".
            assert snapshot.effective().access_emit == "off"
            assert stack.current().access_emit == "errors_only"
        assert snapshot.effective().access_emit == "off"


def test_stacks_are_per_server_not_process_singletons():
    one = PolicyStack()
    two = PolicyStack()
    with one.scope(access_emit="off"):
        assert one.current().access_emit == "off"
        # The other Server is untouched.
        assert two.current().access_emit == "on"
    assert ServerState().policy_stack is not ServerState().policy_stack


def test_config_baseline_merges_into_effective_view():
    async def app(scope, receive, send):  # pragma: no cover
        return None

    config = Config(app=app, access_log=False, log_config=None)
    stack = PolicyStack()
    stack.bind_config(config)
    assert stack.current().access_emit == "off"
    # An explicit pushed layer still wins over the baseline.
    with stack.scope(access_emit="errors_only"):
        assert stack.current().access_emit == "errors_only"


def test_invalid_dimension_values_are_rejected():
    with pytest.raises(ValueError):
        PolicyLayer(lifespan_fault="nope")
    with pytest.raises(ValueError):
        PolicyLayer(access_emit="sometimes")
    with pytest.raises(ValueError):
        PolicyLayer(conn_half_close="halfway")


def test_frame_scope_isolates_concurrent_tasks_and_nests():
    stack = PolicyStack()
    assert stack.current().access_emit == "on"
    with frame_scope(access_emit="off"):
        assert stack.current().access_emit == "off"
        with frame_scope(access_emit="errors_only"):
            assert stack.current().access_emit == "errors_only"
        assert stack.current().access_emit == "off"
    assert stack.current().access_emit == "on"

    async def worker(mode: str) -> str:
        with frame_scope(access_emit=mode):
            await asyncio.sleep(0)
            return stack.current().access_emit

    async def scenario():
        left, right = await asyncio.gather(worker("off"), worker("errors_only"))
        return left, right

    assert _run(scenario()) == ("off", "errors_only")


# --------------------------------------------------------------------------- #
# Lifespan fault policy
# --------------------------------------------------------------------------- #


def test_suppress_startup_continues_without_touching_config_enum():
    async def app(scope, receive, send):
        message = await receive()
        assert message["type"] == "lifespan.startup"
        await send({"type": "lifespan.startup.failed", "message": "the lifespan event failed"})

    async def run():
        config = Config(app=app, lifespan="on", log_config=None)
        lifespan = LifespanOn(config)
        lifespan.bind_policy_stack(PolicyStack())

        with lifespan._policy_stack.scope(lifespan_fault="suppress_startup"):
            await lifespan.startup()

        assert lifespan.startup_failed is True
        # The server must proceed into the main loop.
        assert lifespan.should_exit is False
        # The process-wide Config enum is untouched.
        assert config.lifespan == "on"
        await lifespan.shutdown()

    _run(run())


def test_pin_startup_failure_still_exits():
    async def app(scope, receive, send):
        await receive()
        await send({"type": "lifespan.startup.failed", "message": "boom"})

    async def run():
        config = Config(app=app, lifespan="on", log_config=None)
        lifespan = LifespanOn(config)
        lifespan.bind_policy_stack(PolicyStack())
        await lifespan.startup()
        assert lifespan.should_exit is True
        assert config.lifespan == "on"
        await lifespan.shutdown()

    _run(run())


def test_unbound_lifespan_keeps_historical_behaviour():
    async def app(scope, receive, send):
        await receive()
        await send({"type": "lifespan.startup.failed", "message": "boom"})

    async def run():
        config = Config(app=app, lifespan="on", log_config=None)
        lifespan = LifespanOn(config)  # no Server / no stack attached
        await lifespan.startup()
        assert lifespan.should_exit is True
        await lifespan.shutdown()

    _run(run())


def test_fail_fast_shutdown_callback_only_fires_under_policy():
    stack = PolicyStack()
    fired: list[int] = []
    stack.request_shutdown = lambda: fired.append(1)

    stack.snapshot().notify_uncaught()
    assert fired == []

    with stack.scope(lifespan_fault="fail_fast_shutdown"):
        snapshot = stack.snapshot()
    # The frame was in flight when the policy was active; its frozen snapshot
    # still triggers after the segment has been popped.
    snapshot.notify_uncaught()
    assert fired == [1]


# --------------------------------------------------------------------------- #
# End-to-end: fail_fast_shutdown still runs the lifespan shutdown callback
# --------------------------------------------------------------------------- #


def test_fail_fast_shutdown_runs_lifespan_shutdown_callback():
    events: list[str] = []

    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    events.append("startup")
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    events.append("shutdown")
                    await send({"type": "lifespan.shutdown.complete"})
                    return
            return

        # A fault after the response headers are out cannot deliver a clean
        # status and is therefore an unrecoverable frame fault.
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-length", b"0")],
            }
        )
        raise RuntimeError("boom after headers")

    async def run():
        config = Config(app=app, lifespan="on", port=0, log_config=None)
        server = Server(config)
        server.server_state.policy_stack.push(lifespan_fault="fail_fast_shutdown")

        serve_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)

        port = server.servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)
        writer.write(b"GET / HTTP/1.1\r\nHost: example.org\r\n\r\n")
        await writer.drain()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(reader.read(1024), 2)
        writer.close()

        await asyncio.wait_for(serve_task, 5)
        assert server.force_exit is False
        assert "startup" in events
        assert "shutdown" in events

    _run(run())


def test_suppress_startup_server_enters_main_loop_and_serves_http():
    async def app(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.failed", "message": "no startup"})
                    return
                return
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [(b"content-length", b"0")],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    async def run():
        config = Config(app=app, lifespan="on", port=0, log_config=None)
        server = Server(config)
        assert config.lifespan == "on"
        server.server_state.policy_stack.push(lifespan_fault="suppress_startup")

        serve_task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        assert server.should_exit is False  # entered the main loop

        port = server.servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)
        writer.write(b"GET / HTTP/1.1\r\nHost: example.org\r\n\r\n")
        await writer.drain()
        data = await asyncio.wait_for(reader.read(128), 5)
        assert b" 204 " in data
        writer.close()

        server.should_exit = True
        await asyncio.wait_for(serve_task, 5)
        assert config.lifespan == "on"  # the Config enum is never rewritten

    _run(run())


def test_server_policy_scope_reaches_real_socket_connections():
    messages: list[str] = []

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"connection", b"close")]})
        await send({"type": "http.response.body", "body": b"ok"})

    async def get(port: int, path: str) -> None:
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 5)
        writer.write(f"GET {path} HTTP/1.1\r\nHost: example.org\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()
        await asyncio.wait_for(reader.read(), 3)
        writer.close()

    async def run():
        config = Config(app=app, lifespan="off", port=0, log_config=None)
        server = Server(config)
        access = logging.getLogger("uvicorn.access")
        handler = Handler()
        access.addHandler(handler)
        access.setLevel(logging.INFO)
        try:
            serve_task = asyncio.create_task(server.serve())
            while not server.started:
                await asyncio.sleep(0.01)
            port = server.servers[0].sockets[0].getsockname()[1]

            await get(port, "/one")
            with server.policy_scope(access_emit="off"):
                await get(port, "/two")
            await get(port, "/three")

            server.should_exit = True
            await asyncio.wait_for(serve_task, 5)
        finally:
            access.removeHandler(handler)

        assert any("/one" in m for m in messages)
        assert not any("/two" in m for m in messages)  # scope suppressed this frame
        assert any("/three" in m for m in messages)  # popped cleanly

    _run(run())
