import asyncio
import http
import logging
import time
import traceback

import websockets.asyncio.server as _server
import websockets.frames

from .msgpack_numpy import Packer, unpackb

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        # Single-client guard: the policy + KV cache is stateful and strictly
        # single-client. Track the currently-served connection so a new one
        # supersedes (and force-closes) any stale/orphaned one instead of two
        # handlers interleaving infer() on the shared cache (frame_st_id /
        # KV corruption — observed with reconnects + a half-open tunnel).
        self._active_ws = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
                ping_timeout=None,
        ) as server:
            logger.info(f"Server listening on {self._host}:{self._port}")
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        # Single-client guard (newest connection wins). Force-close any
        # previous connection so its handler's recv() raises ConnectionClosed
        # and it exits cleanly before this one mutates the shared stateful
        # policy. `reset` reinitializes all state, so newest-wins is safe.
        prev = self._active_ws
        self._active_ws = websocket
        if prev is not None and prev is not websocket:
            logger.warning(
                f"New connection {websocket.remote_address} supersedes active "
                f"connection {prev.remote_address}; force-closing the old one.")
            try:
                await prev.close(
                    code=websockets.frames.CloseCode.GOING_AWAY,
                    reason="superseded by a newer client")
            except Exception:
                pass

        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        try:
            while True:
                try:
                    start_time = time.monotonic()
                    obs = unpackb(await websocket.recv())

                    # If a newer connection took over while we were blocked on
                    # recv, stop before touching the shared cache/frame_st_id.
                    if self._active_ws is not websocket:
                        logger.warning(
                            f"Connection {websocket.remote_address} superseded; "
                            f"dropping message and closing handler.")
                        break

                    infer_time = time.monotonic()
                    action = self._policy.infer(obs)
                    infer_time = time.monotonic() - infer_time

                    action["server_timing"] = {
                        "infer_ms": infer_time * 1000,
                    }
                    if prev_total_time is not None:
                        # We can only record the last total time since we also want to include the send time.
                        action["server_timing"][
                            "prev_total_ms"] = prev_total_time * 1000

                    await websocket.send(packer.pack(action))
                    prev_total_time = time.monotonic() - start_time

                except websockets.ConnectionClosed:
                    logger.info(
                        f"Connection from {websocket.remote_address} closed")
                    break
                except Exception:
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason=
                        "Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            # Only release the slot if a newer connection hasn't already
            # claimed it (avoid clobbering the active connection's tracking).
            if self._active_ws is websocket:
                self._active_ws = None


def _health_check(connection: _server.ServerConnection,
                  request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
