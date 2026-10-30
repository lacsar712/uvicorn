from __future__ import annotations

import asyncio
import contextvars
import http
import logging
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import unquote

import h11
from h11._connection import DEFAULT_MAX_INCOMPLETE_EVENT_SIZE

from uvicorn._types import (
    ASGI3Application,
    ASGIReceiveEvent,
    ASGISendEvent,
    HTTPRequestEvent,
    HTTPResponseBodyEvent,
    HTTPResponseStartEvent,
    HTTPScope,
)
from uvicorn.config import Config
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.policy import PolicySnapshot, access_enabled, protocol_error_access_enabled
from uvicorn.protocols.http.flow_control import CLOSE_HEADER, HIGH_WATER_LIMIT, FlowControl, service_unavailable
from uvicorn.protocols.utils import get_client_addr, get_local_addr, get_path_with_query_string, get_remote_addr, is_ssl
from uvicorn.server import ServerState


def _get_status_phrase(status_code: int) -> bytes:
    try:
        return http.HTTPStatus(status_code).phrase.encode()
    except ValueError:
        return b""


STATUS_PHRASES = {status_code: _get_status_phrase(status_code) for status_code in range(100, 600)}


class H11Protocol(asyncio.Protocol):
    def __init__(
        self,
        config: Config,
        server_state: ServerState,
        app_state: dict[str, Any],
        _loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        if not config.loaded:
            config.load()

        self.config = config
        self.app = config.loaded_app
        self.loop = _loop or asyncio.get_event_loop()
        self.logger = logging.getLogger("uvicorn.error")
        self.access_logger = logging.getLogger("uvicorn.access")
        self.access_log = self.access_logger.hasHandlers()
        self.conn = h11.Connection(
            h11.SERVER,
            config.h11_max_incomplete_event_size
            if config.h11_max_incomplete_event_size is not None
            else DEFAULT_MAX_INCOMPLETE_EVENT_SIZE,
        )
        self.ws_protocol_class = config.ws_protocol_class
        self.root_path = config.root_path
        self.asgi_version = config.asgi_version
        self.limit_concurrency = config.limit_concurrency
        self.app_state = app_state

        # Timeouts
        self.timeout_keep_alive_task: asyncio.TimerHandle | None = None
        self.timeout_keep_alive = config.timeout_keep_alive

        # Shared server state
        self.server_state = server_state
        self.connections = server_state.connections
        self.tasks = server_state.tasks
        self.policy_stack = server_state.policy_stack

        # Per-connection state
        self.transport: asyncio.Transport = None  # type: ignore[assignment]
        self.flow: FlowControl = None  # type: ignore[assignment]
        self.server: tuple[str, int | None] | None = None
        self.client: tuple[str, int] | None = None
        self.scheme: Literal["http", "https"] | None = None

        # Per-request state
        self.scope: HTTPScope = None  # type: ignore[assignment]
        self.headers: list[tuple[bytes, bytes]] = None  # type: ignore[assignment]
        self.cycle: RequestResponseCycle = None  # type: ignore[assignment]

    # Protocol interface
    def connection_made(  # type: ignore[override]
        self, transport: asyncio.Transport
    ) -> None:
        self.connections.add(self)

        self.transport = transport
        self.flow = FlowControl(transport)
        self.server = get_local_addr(transport)
        self.client = get_remote_addr(transport)
        self.scheme = "https" if is_ssl(transport) else "http"

        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = "%s:%d - " % self.client if self.client else ""
            self.logger.log(TRACE_LOG_LEVEL, "%sHTTP connection made", prefix)

    def connection_lost(self, exc: Exception | None) -> None:
        self.connections.discard(self)

        if self.logger.level <= TRACE_LOG_LEVEL:
            prefix = "%s:%d - " % self.client if self.client else ""
            self.logger.log(TRACE_LOG_LEVEL, "%sHTTP connection lost", prefix)

        if self.cycle and not self.cycle.response_complete:
            self.cycle.disconnected = True
        if self.conn.our_state != h11.ERROR:
            event = h11.ConnectionClosed()
            try:
                self.conn.send(event)
            except h11.LocalProtocolError:
                # Premature client disconnect
                pass

        if self.cycle is not None:
            self.cycle.message_event.set()
        if self.flow is not None:
            self.flow.resume_writing()
        if exc is None:
            self.transport.close()
            self._unset_keepalive_if_required()

    def eof_received(self) -> None:
        pass

    def _unset_keepalive_if_required(self) -> None:
        if self.timeout_keep_alive_task is not None:
            self.timeout_keep_alive_task.cancel()
            self.timeout_keep_alive_task = None

    def _get_upgrade(self) -> bytes | None:
        connection = []
        upgrade = None
        for name, value in self.headers:
            if name == b"connection":
                connection = [token.lower().strip() for token in value.split(b",")]
            if name == b"upgrade":
                upgrade = value.lower()
        if b"upgrade" in connection:
            return upgrade
        return None

    def _should_upgrade_to_ws(self) -> bool:
        if self.ws_protocol_class is None:
            return False
        return True

    def _unsupported_upgrade_warning(self) -> None:
        msg = "Unsupported upgrade request."
        self.logger.warning(msg)
        if not self._should_upgrade_to_ws():
            msg = "No supported WebSocket library detected. Please use \"pip install 'uvicorn[standard]'\", or install 'websockets' or 'wsproto' manually."  # noqa: E501
            self.logger.warning(msg)

    def _should_upgrade(self) -> bool:
        upgrade = self._get_upgrade()
        if upgrade == b"websocket" and self._should_upgrade_to_ws():
            return True
        if upgrade is not None:
            self._unsupported_upgrade_warning()
        return False

    def data_received(self, data: bytes) -> None:
        self._unset_keepalive_if_required()

        self.conn.receive_data(data)
        self.handle_events()

    def handle_events(self) -> None:
        while True:
            try:
                event = self.conn.next_event()
            except h11.RemoteProtocolError:
                msg = "Invalid HTTP request received."
                self.logger.warning(msg)
                self._log_protocol_error_access()
                self.send_400_response(msg)
                return

            if event is h11.NEED_DATA:
                break

            elif event is h11.PAUSED:
                # This case can occur in HTTP pipelining, so we need to
                # stop reading any more data, and ensure that at the end
                # of the active request/response cycle we handle any
                # events that have been buffered up.
                self.flow.pause_reading()
                break

            elif isinstance(event, h11.Request):
                # Pipelined HTTP requests and WebSocket upgrades may be processed after the keep-alive timer is armed.
                self._unset_keepalive_if_required()

                self.headers = [(key.lower(), value) for key, value in event.headers]
                raw_path, _, query_string = event.target.partition(b"?")
                path = unquote(raw_path.decode("ascii"))
                full_path = self.root_path + path
                full_raw_path = self.root_path.encode("ascii") + raw_path
                self.scope = {
                    "type": "http",
                    "asgi": {"version": self.asgi_version, "spec_version": "2.3"},
                    "http_version": event.http_version.decode("ascii"),
                    "server": self.server,
                    "client": self.client,
                    "scheme": self.scheme,  # type: ignore[typeddict-item]
                    "method": event.method.decode("ascii"),
                    "root_path": self.root_path,
                    "path": full_path,
                    "raw_path": full_raw_path,
                    "query_string": query_string,
                    "headers": self.headers,
                    "state": self.app_state.copy(),
                }
                if self._should_upgrade():
                    self.handle_websocket_upgrade(event)
                    return

                # Handle 503 responses when 'limit_concurrency' is exceeded.
                if self.limit_concurrency is not None and (
                    len(self.connections) >= self.limit_concurrency or len(self.tasks) >= self.limit_concurrency
                ):
                    app = service_unavailable
                    message = "Exceeded concurrency limit."
                    self.logger.warning(message)
                else:
                    app = self.app

                self.cycle = RequestResponseCycle(
                    scope=self.scope,
                    conn=self.conn,
                    transport=self.transport,
                    flow=self.flow,
                    logger=self.logger,
                    access_logger=self.access_logger,
                    access_log=self.access_log,
                    default_headers=self.server_state.default_headers,
                    message_event=asyncio.Event(),
                    on_response=self.on_response_complete,
                    policy=self.policy_stack.snapshot(),
                )
                if self.config.reset_contextvars:
                    # Opt-in workaround for https://github.com/python/cpython/issues/140947:
                    # asyncio can leak context vars between tasks. Hides context set in the
                    # lifespan or by external instrumentation.
                    task = self.loop.create_task(self.cycle.run_asgi(app), context=contextvars.Context())
                else:
                    task = self.loop.create_task(self.cycle.run_asgi(app))
                task.add_done_callback(self.tasks.discard)
                self.tasks.add(task)

            elif isinstance(event, h11.Data):
                if self.conn.our_state is h11.DONE:
                    continue
                self.cycle.body += event.data
                if len(self.cycle.body) > HIGH_WATER_LIMIT:
                    self.flow.pause_reading()
                self.cycle.message_event.set()

            elif isinstance(event, h11.EndOfMessage):
                draining = self.cycle is not None and self.cycle.drain_before_close
                if self.conn.our_state is h11.DONE and not draining:
                    self.transport.resume_reading()
                    self.conn.start_next_cycle()
                    continue
                self.cycle.more_body = False
                self.cycle.message_event.set()
                if self.conn.their_state == h11.MUST_CLOSE:
                    break

    def handle_websocket_upgrade(self, event: h11.Request) -> None:
        if self.logger.level <= TRACE_LOG_LEVEL:  # pragma: full coverage
            prefix = "%s:%d - " % self.client if self.client else ""
            self.logger.log(TRACE_LOG_LEVEL, "%sUpgrading to WebSocket", prefix)

        self.connections.discard(self)
        output = [event.method, b" ", event.target, b" HTTP/1.1\r\n"]
        for name, value in self.headers:
            output += [name, b": ", value, b"\r\n"]
        output.append(b"\r\n")
        protocol = self.ws_protocol_class(  # type: ignore[call-arg, misc]
            config=self.config,
            server_state=self.server_state,
            app_state=self.app_state,
        )
        protocol.connection_made(self.transport)
        self.transport.set_protocol(protocol)
        protocol.data_received(b"".join(output))

    def _log_protocol_error_access(self) -> None:
        effective = self.policy_stack.current()
        if protocol_error_access_enabled(effective, self.access_log):
            client = "%s:%d" % self.client if self.client else ""
            self.access_logger.info('%s - "<invalid request>" 400', client)

    def send_400_response(self, msg: str) -> None:
        reason = STATUS_PHRASES[400]
        headers: list[tuple[bytes, bytes]] = [
            (b"content-type", b"text/plain; charset=utf-8"),
            (b"connection", b"close"),
        ]
        event = h11.Response(status_code=400, headers=headers, reason=reason)
        output = self.conn.send(event)
        self.transport.write(output)

        output = self.conn.send(event=h11.Data(data=msg.encode("ascii")))
        self.transport.write(output)

        output = self.conn.send(event=h11.EndOfMessage())
        self.transport.write(output)

        self.transport.close()

    def on_response_complete(self) -> None:
        self.server_state.total_requests += 1

        if self.transport.is_closing():
            return

        # Set a short Keep-Alive timeout.
        self._unset_keepalive_if_required()

        # conn_half_close=drain_then_close: keep the read side open just long
        # enough to discard the unread request body, then close. No keep-alive
        # timer is armed and no next request cycle is started.
        if self.cycle is not None and self.cycle.drain_before_close:
            self.flow.resume_reading()
            task = self.loop.create_task(self.cycle.drain_request_then_close())
            task.add_done_callback(self.tasks.discard)
            self.tasks.add(task)
            return

        self.timeout_keep_alive_task = self.loop.call_later(self.timeout_keep_alive, self.timeout_keep_alive_handler)

        # Unpause data reads if needed.
        self.flow.resume_reading()

        # Unblock any pipelined events.
        if self.conn.our_state is h11.DONE and self.conn.their_state is h11.DONE:
            self.conn.start_next_cycle()
            self.handle_events()

    def shutdown(self) -> None:
        """
        Called by the server to commence a graceful shutdown.
        """
        if self.cycle is None or self.cycle.response_complete:
            event = h11.ConnectionClosed()
            self.conn.send(event)
            self.transport.close()
        else:
            self.cycle.keep_alive = False

    def pause_writing(self) -> None:
        """
        Called by the transport when the write buffer exceeds the high water mark.
        """
        self.flow.pause_writing()  # pragma: full coverage

    def resume_writing(self) -> None:
        """
        Called by the transport when the write buffer drops below the low water mark.
        """
        self.flow.resume_writing()  # pragma: full coverage

    def timeout_keep_alive_handler(self) -> None:
        """
        Called on a keep-alive connection if no new data is received after a short
        delay.
        """
        if not self.transport.is_closing():
            event = h11.ConnectionClosed()
            self.conn.send(event)
            self.transport.close()


class RequestResponseCycle:
    def __init__(
        self,
        scope: HTTPScope,
        conn: h11.Connection,
        transport: asyncio.Transport,
        flow: FlowControl,
        logger: logging.Logger,
        access_logger: logging.Logger,
        access_log: bool,
        default_headers: list[tuple[bytes, bytes]],
        message_event: asyncio.Event,
        on_response: Callable[..., None],
        policy: PolicySnapshot,
    ) -> None:
        self.scope = scope
        self.conn = conn
        self.transport = transport
        self.flow = flow
        self.logger = logger
        self.access_logger = access_logger
        self.access_log = access_log
        self.default_headers = default_headers
        self.message_event = message_event
        self.on_response = on_response
        # Frozen for this request frame; later pushes/pops never leak in.
        self.policy = policy

        # Connection state
        self.disconnected = False
        self.keep_alive = True
        self.waiting_for_100_continue = conn.they_are_waiting_for_100_continue

        # Request state
        self.body = bytearray()
        self.more_body = True

        # Response state
        self.response_started = False
        self.response_complete = False

        # Set when conn_half_close=drain_then_close must drain the request
        # body before the connection is closed.
        self.drain_before_close = False

    def _maybe_request_fast_shutdown(self) -> None:
        self.policy.notify_uncaught()

    async def drain_request_then_close(self) -> None:
        """Discard the rest of the request body, then close the connection."""
        try:
            while self.more_body and not self.disconnected:
                self.flow.resume_reading()
                await self.message_event.wait()
                self.message_event.clear()
                self.body = bytearray()
        finally:
            if not self.disconnected and not self.transport.is_closing():
                try:
                    self.conn.send(event=h11.ConnectionClosed())
                except h11.LocalProtocolError:
                    pass
                self.transport.close()

    # ASGI exception wrapper
    async def run_asgi(self, app: ASGI3Application) -> None:
        try:
            try:
                result = await app(  # type: ignore[func-returns-value]
                    self.scope, self.receive, self.send
                )
            except BaseException as exc:
                msg = "Exception in ASGI application\n"
                self.logger.error(msg, exc_info=exc)
                if not self.response_started:
                    await self.send_500_response()
                else:
                    # The response headers are already out, so no clean status
                    # can be delivered: this is an unrecoverable frame fault.
                    self._maybe_request_fast_shutdown()
                    self.transport.close()
            else:
                if result is not None:
                    msg = "ASGI callable should return None, but returned '%s'."
                    self.logger.error(msg, result)
                    self.transport.close()
                elif not self.response_started and not self.disconnected:
                    msg = "ASGI callable returned without starting response."
                    self.logger.error(msg)
                    await self.send_500_response()
                elif not self.response_complete and not self.disconnected:
                    msg = "ASGI callable returned without completing response."
                    self.logger.error(msg)
                    self.transport.close()
        except BaseException as exc:
            # The recovery path itself (e.g. building the 500) failed.
            self.logger.error("Exception while recovering from ASGI application error", exc_info=exc)
            self._maybe_request_fast_shutdown()
            if not self.transport.is_closing():
                self.transport.close()
        finally:
            self.on_response = lambda: None

    async def send_500_response(self) -> None:
        response_start_event: HTTPResponseStartEvent = {
            "type": "http.response.start",
            "status": 500,
            "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"connection", b"close"),
            ],
        }
        await self.send(response_start_event)
        response_body_event: HTTPResponseBodyEvent = {
            "type": "http.response.body",
            "body": b"Internal Server Error",
            "more_body": False,
        }
        await self.send(response_body_event)

    # ASGI interface
    async def send(self, message: ASGISendEvent) -> None:
        if self.flow.write_paused and not self.disconnected:
            await self.flow.drain()  # pragma: full coverage

        if self.disconnected:
            return  # pragma: full coverage

        if not self.response_started:
            # Sending response status line and headers
            if message["type"] != "http.response.start":
                raise RuntimeError(f"Expected ASGI message 'http.response.start', but got '{message['type']}'.")

            self.response_started = True
            self.waiting_for_100_continue = False

            status = message["status"]
            headers = self.default_headers + list(message.get("headers", []))

            if CLOSE_HEADER in self.scope["headers"] and CLOSE_HEADER not in headers:
                headers = headers + [CLOSE_HEADER]

            if access_enabled(self.policy.effective(), status, self.access_log):
                self.access_logger.info(
                    '%s - "%s %s HTTP/%s" %d',
                    get_client_addr(self.scope),
                    self.scope["method"],
                    get_path_with_query_string(self.scope),
                    self.scope["http_version"],
                    status,
                )

            # Write response status line and headers
            reason = STATUS_PHRASES[status]
            response = h11.Response(status_code=status, headers=headers, reason=reason)
            output = self.conn.send(event=response)
            self.transport.write(output)

        elif not self.response_complete:
            # Sending response body
            if message["type"] != "http.response.body":
                raise RuntimeError(f"Expected ASGI message 'http.response.body', but got '{message['type']}'.")

            body = message.get("body", b"")
            more_body = message.get("more_body", False)

            # Write response body
            data = b"" if self.scope["method"] == "HEAD" else body
            output = self.conn.send(event=h11.Data(data=data))
            self.transport.write(output)

            # Handle response completion
            if not more_body:
                self.response_complete = True
                self.message_event.set()
                output = self.conn.send(event=h11.EndOfMessage())
                self.transport.write(output)

        else:
            # Response already sent
            raise RuntimeError(f"Unexpected ASGI message '{message['type']}' sent, after response already completed.")

        if self.response_complete:
            half_close = self.policy.effective().conn_half_close
            must_close_now = self.conn.our_state is h11.MUST_CLOSE or not self.keep_alive or half_close == "force_close"
            if half_close == "drain_then_close" and not must_close_now and self.more_body:
                # The response is already fully written; keep the connection
                # open just long enough to drain the unread request body.
                self.keep_alive = False
                self.drain_before_close = True
            elif must_close_now or half_close == "drain_then_close":
                # force_close reaches here immediately (the frame response is
                # finished, so no in-flight response iterator is torn down).
                self.conn.send(event=h11.ConnectionClosed())
                self.transport.close()
            self.on_response()

    async def receive(self) -> ASGIReceiveEvent:
        if self.waiting_for_100_continue and not self.transport.is_closing():
            headers: list[tuple[str, str]] = []
            event = h11.InformationalResponse(status_code=100, headers=headers, reason="Continue")
            output = self.conn.send(event=event)
            self.transport.write(output)
            self.waiting_for_100_continue = False

        if not self.disconnected and not self.response_complete:
            self.flow.resume_reading()
            await self.message_event.wait()
            self.message_event.clear()

        if self.disconnected or self.response_complete:
            return {"type": "http.disconnect"}

        message: HTTPRequestEvent = {"type": "http.request", "body": bytes(self.body), "more_body": self.more_body}
        self.body = bytearray()
        return message
