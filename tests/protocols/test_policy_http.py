from __future__ import annotations

import logging
from typing import Any

import pytest

from tests.protocols.test_http import (
    FINISH_POST_REQUEST,
    INVALID_REQUEST_TEMPLATE,
    SIMPLE_GET_REQUEST,
    START_POST_REQUEST,
    get_connected_protocol as get_h1_protocol,
)
from tests.response import Response
from uvicorn.policy import PolicyStack
from uvicorn.protocols.http.h11_impl import H11Protocol

try:
    import zttp

    from tests.protocols.test_http2 import H2Client, get_connected_protocol as get_h2_protocol

    have_h2 = hasattr(zttp, "HTTP2")
except ModuleNotFoundError:  # pragma: no cover
    have_h2 = False

skip_without_h2 = pytest.mark.skipif(not have_h2, reason="zttp with HTTP/2 support is not installed")

pytestmark = pytest.mark.anyio


def _capture_access(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="uvicorn.access")
    caplog.set_level(logging.INFO, logger="uvicorn.error")
    logging.getLogger("uvicorn.access").propagate = True


def _access_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.message for record in caplog.records if record.name == "uvicorn.access"]


def _error_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.message for record in caplog.records if record.name == "uvicorn.error"]


def _get(path: str) -> bytes:
    return b"\r\n".join([f"GET {path} HTTP/1.1".encode(), b"Host: example.org", b"", b""])


async def _raise_before_response(scope: Any, receive: Any, send: Any) -> None:
    raise RuntimeError("boom")


async def _raise_after_start(scope: Any, receive: Any, send: Any) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"0")]})
    raise RuntimeError("boom after headers")


INVALID_REQUEST = INVALID_REQUEST_TEMPLATE % b"GET / HTTP1.1"


# --------------------------------------------------------------------------- #
# H11: access_emit
# --------------------------------------------------------------------------- #


async def test_h11_access_on_baseline_emits(caplog: pytest.LogCaptureFixture):
    _capture_access(caplog)
    protocol = get_h1_protocol(Response("x"), H11Protocol, log_config=None)
    protocol.data_received(_get("/"))
    await protocol.loop.run_one()
    assert any('"GET / HTTP/1.1" 200' in line for line in _access_lines(caplog))


async def test_h11_access_off_suppresses_access_but_keeps_error_log(caplog: pytest.LogCaptureFixture):
    _capture_access(caplog)
    protocol = get_h1_protocol(_raise_after_start, H11Protocol, log_config=None)
    with protocol.policy_stack.scope(access_emit="off"):
        protocol.data_received(_get("/"))
        await protocol.loop.run_one()
    assert _access_lines(caplog) == []
    assert any("Exception in ASGI application" in line for line in _error_lines(caplog))


async def test_h11_errors_only_emits_for_5xx_and_protocol_error(caplog: pytest.LogCaptureFixture):
    _capture_access(caplog)

    ok = get_h1_protocol(Response("x"), H11Protocol, log_config=None)
    with ok.policy_stack.scope(access_emit="errors_only"):
        ok.data_received(_get("/ok"))
        await ok.loop.run_one()
    assert _access_lines(caplog) == []

    broken = get_h1_protocol(_raise_before_response, H11Protocol, log_config=None)
    with broken.policy_stack.scope(access_emit="errors_only"):
        broken.data_received(_get("/boom"))
        await broken.loop.run_one()
    assert any('"GET /boom HTTP/1.1" 500' in line for line in _access_lines(caplog))

    bad = get_h1_protocol(Response("x"), H11Protocol, log_config=None)
    with bad.policy_stack.scope(access_emit="errors_only"):
        bad.data_received(INVALID_REQUEST)
    assert any("400" in line for line in _access_lines(caplog))


async def test_h11_two_connections_with_different_policy_do_not_cross_frames(
    caplog: pytest.LogCaptureFixture,
):
    _capture_access(caplog)
    protocol_a = get_h1_protocol(Response("a"), H11Protocol, log_config=None)
    protocol_b = get_h1_protocol(Response("b"), H11Protocol, log_config=None)
    # Same per-Server stack for both connections.
    protocol_b.policy_stack = protocol_a.policy_stack
    stack = protocol_a.policy_stack

    with stack.scope(access_emit="off"):
        # Connection A's frame starts while "off" is active.
        protocol_a.data_received(_get("/a"))
    # The segment is popped before connection B's frame starts.
    protocol_b.data_received(_get("/b"))

    await protocol_a.loop.run_one()
    await protocol_b.loop.run_one()

    lines = _access_lines(caplog)
    assert any("/b" in line for line in lines)
    assert not any("/a" in line for line in lines)


# --------------------------------------------------------------------------- #
# H11: conn_half_close
# --------------------------------------------------------------------------- #


async def test_h11_force_close_closes_after_response_despite_keep_alive():
    protocol = get_h1_protocol(Response("x"), H11Protocol)
    with protocol.policy_stack.scope(conn_half_close="force_close"):
        protocol.data_received(SIMPLE_GET_REQUEST)
        await protocol.loop.run_one()
    assert protocol.transport.is_closing()


async def test_h11_pin_keeps_keep_alive():
    protocol = get_h1_protocol(Response("x"), H11Protocol)
    protocol.data_received(SIMPLE_GET_REQUEST)
    await protocol.loop.run_one()
    assert not protocol.transport.is_closing()


async def test_h11_drain_then_close_waits_for_unread_body_then_closes():
    # The app responds without consuming the POST body.
    protocol = get_h1_protocol(Response("x"), H11Protocol)
    with protocol.policy_stack.scope(conn_half_close="drain_then_close"):
        protocol.data_received(START_POST_REQUEST)
        await protocol.loop.run_one()
        # Response is written but the request body has not arrived yet.
        assert not protocol.transport.is_closing()
        assert protocol.cycle.drain_before_close is True

        protocol.data_received(FINISH_POST_REQUEST)
        await protocol.loop.run_one()
        assert protocol.transport.is_closing()


# --------------------------------------------------------------------------- #
# H11: fail_fast_shutdown trigger
# --------------------------------------------------------------------------- #


async def test_h11_fail_fast_triggers_shutdown_on_unrecoverable_frame_fault():
    fired: list[int] = []
    protocol = get_h1_protocol(_raise_after_start, H11Protocol, log_config=None)
    protocol.policy_stack = PolicyStack()
    protocol.policy_stack.request_shutdown = lambda: fired.append(1)

    with protocol.policy_stack.scope(lifespan_fault="fail_fast_shutdown"):
        protocol.data_received(_get("/"))
        await protocol.loop.run_one()
    assert fired == [1]


async def test_h11_pin_does_not_trigger_shutdown_on_frame_fault():
    fired: list[int] = []
    protocol = get_h1_protocol(_raise_after_start, H11Protocol, log_config=None)
    protocol.policy_stack.request_shutdown = lambda: fired.append(1)

    protocol.data_received(_get("/"))
    await protocol.loop.run_one()
    assert fired == []


# --------------------------------------------------------------------------- #
# H2 (zttp)
# --------------------------------------------------------------------------- #


@skip_without_h2
async def test_h2_access_off_suppresses_access(caplog: pytest.LogCaptureFixture):
    _capture_access(caplog)
    protocol = get_h2_protocol(Response("x"), log_config=None)
    client = H2Client()
    with protocol.policy_stack.scope(access_emit="off"):
        client.request(b"GET", b"/off")
        protocol.data_received(client.data_to_send())
        await protocol.loop.run_one()
    assert _access_lines(caplog) == []
    assert protocol.transport.buffer  # response still produced


@skip_without_h2
async def test_h2_errors_only_emits_for_5xx_not_2xx(caplog: pytest.LogCaptureFixture):
    _capture_access(caplog)

    ok = get_h2_protocol(Response("x"), log_config=None)
    client_ok = H2Client()
    with ok.policy_stack.scope(access_emit="errors_only"):
        client_ok.request(b"GET", b"/ok")
        ok.data_received(client_ok.data_to_send())
        await ok.loop.run_one()
    assert _access_lines(caplog) == []

    broken = get_h2_protocol(_raise_before_response, log_config=None)
    client_bad = H2Client()
    with broken.policy_stack.scope(access_emit="errors_only"):
        client_bad.request(b"GET", b"/boom")
        broken.data_received(client_bad.data_to_send())
        await broken.loop.run_one()
    assert any("500" in line for line in _access_lines(caplog))


@skip_without_h2
async def test_h2_two_streams_do_not_cross_frames(caplog: pytest.LogCaptureFixture):
    _capture_access(caplog)
    protocol_a = get_h2_protocol(Response("a"), log_config=None)
    protocol_b = get_h2_protocol(Response("b"), log_config=None)
    protocol_b.policy_stack = protocol_a.policy_stack
    stack = protocol_a.policy_stack

    client_a = H2Client()
    client_b = H2Client()
    with stack.scope(access_emit="off"):
        client_a.request(b"GET", b"/a")
        protocol_a.data_received(client_a.data_to_send())
    client_b.request(b"GET", b"/b")
    protocol_b.data_received(client_b.data_to_send())

    await protocol_a.loop.run_one()
    await protocol_b.loop.run_one()

    lines = _access_lines(caplog)
    assert any("/b" in line for line in lines)
    assert not any("/a" in line for line in lines)


@skip_without_h2
async def test_h2_force_close_sends_goaway_and_closes_after_frame():
    protocol = get_h2_protocol(Response("x"))
    client = H2Client()
    with protocol.policy_stack.scope(conn_half_close="force_close"):
        client.request(b"GET", b"/")
        protocol.data_received(client.data_to_send())
        await protocol.loop.run_one()
    assert protocol.shutdown_requested is True
    assert protocol.transport.is_closing()


@skip_without_h2
async def test_h2_pin_keeps_connection_open():
    protocol = get_h2_protocol(Response("x"))
    client = H2Client()
    client.request(b"GET", b"/")
    protocol.data_received(client.data_to_send())
    await protocol.loop.run_one()
    assert not protocol.transport.is_closing()


@skip_without_h2
async def test_h2_drain_then_close_waits_for_request_body():
    protocol = get_h2_protocol(Response("x"))
    client = H2Client()
    with protocol.policy_stack.scope(conn_half_close="drain_then_close"):
        stream = client.request(b"POST", b"/upload", end=False)
        protocol.data_received(client.data_to_send())
        await protocol.loop.run_one()
        # Response is done, but the request body upload is not.
        assert not protocol.transport.is_closing()
        assert 1 in protocol._draining

        stream.send_data(b"0123456789")
        stream.end_message()
        protocol.data_received(client.data_to_send())

    assert protocol.transport.is_closing()
