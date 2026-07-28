# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench server`` CLI command.

Covers:
- Sub-command registration under ``lmcache bench``
- Argument registration and defaults
- Pure helper functions (_build_token_ids, _make_key, _query_checksum)
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import cast
import _thread
import argparse
import itertools
import json
import threading

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers
from lmcache.cli.commands.bench.server_bench import runner as sv_runner
from lmcache.cli.commands.bench.server_bench.helpers import (
    RequestResult,
    ServerTaintedError,
    ServerTaintedInterrupt,
    _allocate_kv_cache,
    _build_token_ids,
    _make_key,
    _poll_prefetch_status,
    _process_request,
    _query_checksum,
    _send_lookup,
    _send_unregister_kv_cache,
)
from lmcache.cli.commands.bench.server_bench.runner import (
    WorkerRuntime,
    _worker_seq_numbers,
    run_concurrent_pairs,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType

# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def cmd() -> BenchCommand:
    return BenchCommand()


@pytest.fixture
def parser(cmd: BenchCommand) -> argparse.ArgumentParser:
    """Parser with ``bench server`` subcommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


# ------------------------------------------------------------------ #
#  Command metadata
# ------------------------------------------------------------------ #


class TestCommandMetadata:
    def test_name(self, cmd: BenchCommand) -> None:
        assert cmd.name() == "bench"

    def test_help(self, cmd: BenchCommand) -> None:
        assert "benchmark" in cmd.help().lower()

    def test_server_helpers_live_under_server_bench_package(self) -> None:
        """Helpers backing ``bench server`` must live inside the
        ``server_bench`` sub-package, mirroring the engine / l2 layout.
        """
        # First Party
        from lmcache.cli.commands.bench.server_bench import command as sv_cmd
        from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers

        assert sv_cmd.__name__ == ("lmcache.cli.commands.bench.server_bench.command")
        assert sv_helpers.__name__ == (
            "lmcache.cli.commands.bench.server_bench.helpers"
        )
        # Public command surface mirrors the sibling subpackages.
        assert callable(sv_cmd.add_server_arguments)
        assert callable(sv_cmd.run_server_bench)


# ------------------------------------------------------------------ #
#  Argument registration
# ------------------------------------------------------------------ #


class TestCommandArguments:
    def test_registers_subcommand(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "server"])
        assert hasattr(args, "func")
        assert args.bench_target == "server"

    def test_default_values(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "server"])
        assert args.rpc_url == "tcp://localhost:5555"
        assert args.mode == "gpu"
        assert args.num_tokens == 512
        assert args.num_blocks == 1024
        assert args.block_size == 16
        assert args.start == 0
        assert args.end is None
        assert args.interval == 0.5
        assert args.url == "http://localhost:8080"

    def test_custom_values(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--rpc-url",
                "tcp://host:9999",
                "--num-tokens",
                "256",
                "--num-blocks",
                "512",
                "--block-size",
                "8",
                "--start",
                "5",
                "--end",
                "10",
                "--interval",
                "1.0",
                "--url",
                "http://other:9090",
            ],
        )
        assert args.rpc_url == "tcp://host:9999"
        assert args.num_tokens == 256
        assert args.num_blocks == 512
        assert args.block_size == 8
        assert args.start == 5
        assert args.end == 10
        assert args.interval == 1.0
        assert args.url == "http://other:9090"

    def test_kvcache_shape_spec_default(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(["bench", "server"])
        assert "float16" in args.kvcache_shape_spec

    def test_kvcache_shape_spec_custom(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--kvcache-shape-spec",
                "(2,512,8,4,64):bfloat16:16",
            ],
        )
        assert args.kvcache_shape_spec == ("(2,512,8,4,64):bfloat16:16")


# ------------------------------------------------------------------ #
#  _build_token_ids
# ------------------------------------------------------------------ #


class TestBuildTokenIds:
    def test_basic(self):
        ids = _build_token_ids(seq_no=7, num_tokens=3)
        assert ids[0] == 7
        assert len(ids) == 4  # seq_no + 3 hello tokens
        # All remaining tokens should be the hello token
        assert all(t == 9906 for t in ids[1:])

    def test_zero_tokens(self):
        ids = _build_token_ids(seq_no=0, num_tokens=0)
        assert ids == (0,)

    def test_different_seq_no(self):
        ids1 = _build_token_ids(seq_no=1, num_tokens=2)
        ids2 = _build_token_ids(seq_no=2, num_tokens=2)
        assert ids1[0] != ids2[0]
        assert ids1[1:] == ids2[1:]


# ------------------------------------------------------------------ #
#  _make_key
# ------------------------------------------------------------------ #


class TestMakeKey:
    def test_basic_key(self):
        token_ids = (0, 9906, 9906)
        key = _make_key(
            token_ids,
            request_id="req-0-cold",
        )
        assert key.model_name == "test-model"
        assert key.world_size == 1
        assert key.worker_id is None
        assert key.token_ids == token_ids
        assert key.start == 0
        assert key.end == len(token_ids)
        assert key.request_id == "req-0-cold"

    def test_custom_start_end(self):
        token_ids = (0, 9906, 9906, 9906, 9906)
        key = _make_key(
            token_ids,
            request_id="req-1-warm",
            start=2,
            end=4,
        )
        assert key.start == 2
        assert key.end == 4

    def test_worker_id(self):
        token_ids = (0, 9906)
        key = _make_key(
            token_ids,
            request_id="req-0-cold",
            worker_id=0,
        )
        assert key.worker_id == 0


# ------------------------------------------------------------------ #
#  _query_checksum
# ------------------------------------------------------------------ #


class _ChecksumHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that records the POST body and returns fake checksums.

    Mirrors the MP server's ``POST /cache/checksums`` (the old ``GET
    /kvcache/check`` was removed). The received JSON is stored on the server so
    the test can assert the request shape ``_query_checksum`` sends.
    """

    def do_POST(self):
        if "/cache/checksums" not in self.path:
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode())
        self.server.received_payloads.append(payload)
        body = json.dumps(
            {
                "status": "success",
                "chunk_checksums": ["a" * 32, "b" * 32],
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # suppress logs


class TestQueryChecksum:
    @pytest.fixture(autouse=True)
    def _start_server(self):
        """Start a tiny HTTP server for the test."""
        self.server = HTTPServer(
            ("127.0.0.1", 0),
            _ChecksumHandler,
        )
        self.server.received_payloads = []
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(
            target=self.server.serve_forever,
        )
        self.thread.daemon = True
        self.thread.start()
        yield
        self.server.shutdown()

    def test_success(self):
        base = "http://127.0.0.1:%d" % self.port
        result = _query_checksum(
            base,
            block_offset=0,
            num_blocks=2,
            block_size=2,
            chunk_size=2,
        )
        assert result is not None
        assert len(result) == 2
        assert result[0] == "a" * 32
        # The POST body matches the MP /cache/checksums contract: block-native
        # ids and a block-level chunk_size (token chunk_size 2 / block_size 2).
        assert len(self.server.received_payloads) == 1
        sent = self.server.received_payloads[0]
        assert sent["block_ids"] == [0, 1]
        assert sent["chunk_size"] == 1
        assert sent["layerwise"] is False

    def test_unreachable_returns_none(self):
        result = _query_checksum(
            "http://127.0.0.1:1",
            block_offset=0,
            num_blocks=2,
            block_size=2,
            chunk_size=2,
        )
        assert result is None


# ------------------------------------------------------------------ #
#  ROUTER endpoint fixture                                             #
# ------------------------------------------------------------------ #


@pytest.fixture
def router_endpoint() -> str:
    """Allocate an ephemeral inproc/tcp endpoint for the ROUTER."""
    # Use tcp with port=0 so the OS assigns a free port.
    ctx = zmq.Context.instance()
    probe = ctx.socket(zmq.ROUTER)
    probe.bind("tcp://127.0.0.1:0")
    endpoint = probe.getsockopt_string(zmq.LAST_ENDPOINT)
    probe.close(linger=0)
    return endpoint


# ------------------------------------------------------------------ #
#  _allocate_kv_cache (dtype branching)
# ------------------------------------------------------------------ #


class TestAllocateKVCache:
    """Regression tests for ``_allocate_kv_cache`` dtype handling.

    ``torch.randn`` only supports floating-point dtypes, so integer
    dtypes in ``DTYPE_MAP`` (e.g. ``uint8`` used by FP8 quantized
    layouts) must fall back to ``torch.randint`` -- see Bugbot
    #3147565172.
    """

    @staticmethod
    def _alloc(dtype: torch.dtype) -> list[torch.Tensor]:
        return _allocate_kv_cache(
            num_layers=1,
            num_heads=2,
            head_size=4,
            num_blocks=2,
            block_size=2,
            dtype=dtype,
            device="cpu",
            kv_size=2,
        )

    @pytest.mark.parametrize(
        "dtype",
        [torch.float16, torch.float32, torch.bfloat16],
    )
    def test_floating_point_dtype(self, dtype: torch.dtype) -> None:
        tensors = self._alloc(dtype)
        assert len(tensors) == 1
        assert tensors[0].dtype == dtype
        assert tensors[0].shape == (2, 2, 2, 2, 4)

    def test_uint8_dtype_uses_randint(self) -> None:
        """Regression: ``torch.randn`` crashes with integer dtypes."""
        tensors = self._alloc(torch.uint8)
        assert len(tensors) == 1
        assert tensors[0].dtype == torch.uint8
        assert tensors[0].shape == (2, 2, 2, 2, 4)

    def test_groups_honour_per_group_shape_and_dtype(self) -> None:
        """Multi-group spec must allocate per-layer shape / dtype.

        Regression for Bugbot #3150738055: previously every layer was
        allocated with the *first* group's ``nh`` / ``hs`` / ``dtype``
        (and the total ``num_layers`` from the sum), silently producing
        wrong tensors for layers in later groups.
        """
        # Standard
        from types import SimpleNamespace

        # First Party
        from lmcache.v1.kv_layer_groups import KVLayerGroupInfo

        # Group A: 3 layers of (2, 2, 2, 8, 16), float16
        # Group B: 2 layers of (1, 2, 2, 4, 32), bfloat16
        # (NB / BS are intentionally identical — that's a hard
        # requirement of paged KV, enforced in CLI execute().)
        group_a = KVLayerGroupInfo(
            layer_indices=[0, 1, 2],
            shape_desc=SimpleNamespace(kv_size=2, nb=2, bs=2, nh=8, hs=16, nl=3),
            dtype=torch.float16,
        )
        group_b = KVLayerGroupInfo(
            layer_indices=[3, 4],
            shape_desc=SimpleNamespace(kv_size=1, nb=2, bs=2, nh=4, hs=32, nl=2),
            dtype=torch.bfloat16,
        )
        tensors = _allocate_kv_cache(
            device="cpu",
            groups=[group_a, group_b],
        )
        assert len(tensors) == 5
        for t in tensors[:3]:
            assert t.shape == (2, 2, 2, 8, 16)
            assert t.dtype == torch.float16
        for t in tensors[3:]:
            assert t.shape == (1, 2, 2, 4, 32)
            assert t.dtype == torch.bfloat16


# ------------------------------------------------------------------ #
#  _send_lookup / _poll_prefetch_status (protocol regression)          #
# ------------------------------------------------------------------ #


class _LookupRouter:
    """Fake ROUTER implementing the LOOKUP / QUERY_PREFETCH_STATUS
    subset of the MP server protocol.

    * ``LOOKUP`` replies with **no payload** (void response) — the
      real server-side handler returns ``None``. Regression for a
      bug where the client treated the empty frame list as a
      timeout and printed ``LOOKUP timeout``.
    * ``QUERY_PREFETCH_STATUS`` accepts a ``request_id`` (str) and
      returns ``None`` on the first N polls, then a fixed chunk
      count — exercising both the in-progress and done branches.
    """

    def __init__(
        self,
        endpoint: str,
        in_progress_polls: int = 1,
        hit_chunks: int = 3,
    ) -> None:
        self._endpoint = endpoint
        self._in_progress_left = in_progress_polls
        self._hit_chunks = hit_chunks
        self.last_query_request_id: str | None = None
        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(endpoint)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._router.close(linger=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._router.poll(100, zmq.POLLIN):
                continue
            frames = self._router.recv_multipart()
            identity, uid_f, type_f, *payload = frames
            req_type = msgspec.msgpack.decode(type_f, type=RequestType)
            if req_type == RequestType.LOOKUP:
                # Void reply: no payload frame.
                self._router.send_multipart([identity, uid_f, type_f])
            elif req_type == RequestType.QUERY_PREFETCH_STATUS:
                req_id = msgspec.msgpack.decode(payload[0], type=str)
                self.last_query_request_id = req_id
                if self._in_progress_left > 0:
                    self._in_progress_left -= 1
                    body = msgspec.msgpack.encode(None)
                else:
                    body = msgspec.msgpack.encode(self._hit_chunks)
                self._router.send_multipart([identity, uid_f, type_f, body])


class TestLookupProtocol:
    def _make_client(self, endpoint: str) -> MessageQueueClient:
        ctx = zmq.Context.instance()
        return MessageQueueClient(endpoint, ctx)

    def test_send_lookup_void_reply_is_success(
        self,
        router_endpoint: str,
    ) -> None:
        """LOOKUP handler returns None (void) — must not be timeout."""
        router = _LookupRouter(router_endpoint)
        router.start()
        try:
            client = self._make_client(router_endpoint)
            key = _make_key((1, 9906, 9906), request_id="req-void")
            assert _send_lookup(client, key) is True
            client.close()
        finally:
            router.stop()

    def test_poll_prefetch_status_uses_request_id(
        self,
        router_endpoint: str,
    ) -> None:
        """QUERY_PREFETCH_STATUS payload is keyed by request_id str."""
        router = _LookupRouter(
            router_endpoint,
            in_progress_polls=2,
            hit_chunks=5,
        )
        router.start()
        try:
            client = self._make_client(router_endpoint)
            hit = _poll_prefetch_status(
                client,
                "req-42",
                max_polls=10,
                poll_interval=0.0,
            )
            assert hit == 5
            assert router.last_query_request_id == "req-42"
            client.close()
        finally:
            router.stop()


# ------------------------------------------------------------------ #
#  _send_unregister_kv_cache (deregister on shutdown)                  #
# ------------------------------------------------------------------ #


class _UnregisterRouter:
    """Fake ROUTER that records UNREGISTER requests and replies void.

    Both ``UNREGISTER_KV_CACHE`` and
    ``UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` carry a single
    ``instance_id`` payload and return ``None`` (void). This fake
    records the request type and decoded ``instance_id`` of the last
    UNREGISTER it saw so the test can assert the bench sends the
    correct protocol for each transfer mode.
    """

    def __init__(self, endpoint: str) -> None:
        self.last_request_type: RequestType | None = None
        self.last_instance_id: int | None = None
        self._ctx = zmq.Context.instance()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(endpoint)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._router.close(linger=0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._router.poll(100, zmq.POLLIN):
                continue
            frames = self._router.recv_multipart()
            identity, uid_f, type_f, *payload = frames
            req_type = msgspec.msgpack.decode(type_f, type=RequestType)
            if req_type in (
                RequestType.UNREGISTER_KV_CACHE,
                RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            ):
                self.last_request_type = req_type
                self.last_instance_id = msgspec.msgpack.decode(payload[0], type=int)
                # Void reply: no payload frame.
                self._router.send_multipart([identity, uid_f, type_f])


class TestUnregisterKVCache:
    def _make_client(self, endpoint: str) -> MessageQueueClient:
        ctx = zmq.Context.instance()
        return MessageQueueClient(endpoint, ctx)

    def test_handle_mode_sends_unregister_kv_cache(
        self,
        router_endpoint: str,
    ) -> None:
        """Handle mode uses the GPU/SHM ``UNREGISTER_KV_CACHE`` protocol."""
        router = _UnregisterRouter(router_endpoint)
        router.start()
        try:
            client = self._make_client(router_endpoint)
            assert (
                _send_unregister_kv_cache(client, instance_id=7, use_handle=True)
                is True
            )
            assert router.last_request_type == RequestType.UNREGISTER_KV_CACHE
            assert router.last_instance_id == 7
            client.close()
        finally:
            router.stop()

    def test_data_mode_sends_engine_driven_unregister(
        self,
        router_endpoint: str,
    ) -> None:
        """Data mode uses the engine-driven context unregister protocol."""
        router = _UnregisterRouter(router_endpoint)
        router.start()
        try:
            client = self._make_client(router_endpoint)
            assert (
                _send_unregister_kv_cache(client, instance_id=0, use_handle=False)
                is True
            )
            assert (
                router.last_request_type
                == RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
            )
            assert router.last_instance_id == 0
            client.close()
        finally:
            router.stop()


# ------------------------------------------------------------------ #
#  _process_request lifecycle matrix (fail-close + submit-then-unknown)
# ------------------------------------------------------------------ #

# Stand-in client: every RPC is injected by patching ``sv_helpers._call``,
# so nothing is ever called on the client object itself.
_DUMMY_CLIENT = cast(MessageQueueClient, object())


def _dispatching_call(behavior, calls=None):
    """Build a ``_call`` replacement that dispatches on request type.

    ``behavior`` maps ``RequestType`` -> value | callable(payloads) -> value.
    A callable may raise to inject an exception / Ctrl-C at the real RPC
    wait point. The sentinel ``sv_helpers._TIMEOUT`` simulates an RPC
    timeout; unlisted types reply void (``None``). Every request type is
    appended to ``calls`` (when given) for issued-operation assertions.
    """

    def _fake_call(client, request_type, payloads, timeout_s=10.0):
        if calls is not None:
            calls.append(request_type)
        action = behavior.get(request_type)
        if callable(action):
            return action(payloads)
        return action

    return _fake_call


def _pair_kwargs(**overrides):
    """Baseline kwargs driving _process_request as a handle-mode pair request.

    511 + 1 seq token = 512 tokens = 2 chunks of 256, so a poll hit of 1
    exercises both the RETRIEVE (hit) and the STORE (miss) leg.
    """
    base = dict(
        num_tokens=511,
        chunk_size=256,
        pass_label="cold",
        http_base="",
        block_size=16,
        total_blocks=1024,
        num_engine_group_infos=1,
        use_gpu=False,
        use_handle=True,
        client_tensors=None,
        server_pool=None,
    )
    base.update(overrides)
    return base


def _success_behavior():
    """A fully successful handle-mode pair request; rows override one stage."""
    return {
        RequestType.LOOKUP: None,
        RequestType.QUERY_PREFETCH_STATUS: 1,
        RequestType.RETRIEVE: (0, True),
        RequestType.STORE: (0, True),
        RequestType.END_SESSION: None,
    }


def _raise(exc_type):
    """A ``_call`` action that raises *exc_type* at the RPC wait point."""

    def _action(_payloads):
        raise exc_type

    return _action


class TestProcessRequestLifecycle:
    """The single-request contract: never a confident-but-wrong result.

    Every stateful RPC is submit-then-unknown, so a failure after LOOKUP is
    submitted invalidates the run (``failure`` set) and, when the server may
    hold indeterminate state, taints it (``server_tainted``); a body error or
    a cleanup that cannot be acknowledged is raised as ``ServerTaintedError``
    / ``ServerTaintedInterrupt``. ``None`` is returned only for a legal skip
    before any RPC is submitted.
    """

    def test_legal_skip_returns_none_without_any_rpc(self, monkeypatch) -> None:
        calls: list = []
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call({}, calls))
        # 0 tokens -> a single seq token -> fewer than one full chunk.
        result = _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs(num_tokens=0))
        assert result is None
        assert calls == []

    def test_success_is_valid_and_untainted(self, monkeypatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            sv_helpers, "_call", _dispatching_call(_success_behavior(), calls)
        )
        result = _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        assert result is not None
        assert result.failure == ""
        assert result.server_tainted is False
        # The pair request retrieves the hit, stores the miss, ends session.
        assert RequestType.RETRIEVE in calls
        assert RequestType.STORE in calls
        assert RequestType.END_SESSION in calls

    @pytest.mark.parametrize(
        "inject, failure_substr, tainted",
        [
            ({RequestType.LOOKUP: sv_helpers._TIMEOUT}, "LOOKUP timeout", True),
            (
                {RequestType.QUERY_PREFETCH_STATUS: sv_helpers._TIMEOUT},
                "prefetch status poll failed",
                True,
            ),
            ({RequestType.RETRIEVE: (0, False)}, "RETRIEVE retrieve_failed", False),
            ({RequestType.STORE: (0, False)}, "STORE store_failed", False),
            (
                {RequestType.END_SESSION: sv_helpers._TIMEOUT},
                "END_SESSION timeout",
                True,
            ),
        ],
        ids=[
            "lookup_timeout",
            "poll_failure",
            "retrieve_failure",
            "store_failure",
            "end_session_timeout",
        ],
    )
    def test_failure_rows_invalidate_and_maybe_taint(
        self, monkeypatch, inject, failure_substr, tainted
    ) -> None:
        calls: list = []
        behavior = _success_behavior()
        behavior.update(inject)
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        # A post-LOOKUP failure is a RequestResult with a reason -- never a
        # bare None and never a silent success.
        assert result is not None
        assert failure_substr in result.failure
        assert result.server_tainted is tainted
        # END_SESSION cleanup is attempted on every post-LOOKUP exit.
        assert RequestType.END_SESSION in calls

    @pytest.mark.parametrize(
        "inject, exc_type",
        [
            ({RequestType.END_SESSION: _raise(RuntimeError)}, ServerTaintedError),
            (
                {RequestType.END_SESSION: _raise(KeyboardInterrupt)},
                ServerTaintedInterrupt,
            ),
            (
                {RequestType.QUERY_PREFETCH_STATUS: _raise(RuntimeError)},
                ServerTaintedError,
            ),
            (
                {RequestType.QUERY_PREFETCH_STATUS: _raise(KeyboardInterrupt)},
                ServerTaintedInterrupt,
            ),
        ],
        ids=[
            "end_session_exception",
            "end_session_ctrl_c",
            "body_exception_mid_poll",
            "body_ctrl_c_mid_poll",
        ],
    )
    def test_taint_rows_raise_through_producer(
        self, monkeypatch, inject, exc_type
    ) -> None:
        behavior = _success_behavior()
        behavior.update(inject)
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior))
        with pytest.raises(exc_type):
            _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())

    def test_ctrl_c_taint_keeps_interrupt_semantics(self, monkeypatch) -> None:
        # ServerTaintedInterrupt subclasses KeyboardInterrupt, so the run
        # still exits 130 while flagging the server unsafe to reuse; the
        # original interrupt is chained.
        behavior = _success_behavior()
        behavior[RequestType.END_SESSION] = _raise(KeyboardInterrupt)
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior))
        with pytest.raises(ServerTaintedInterrupt) as ei:
            _process_request(_DUMMY_CLIENT, 0, **_pair_kwargs())
        assert isinstance(ei.value, KeyboardInterrupt)
        assert isinstance(ei.value.__cause__, KeyboardInterrupt)


# ------------------------------------------------------------------ #
#  Concurrent runner (barrier / band / verdict / teardown)
# ------------------------------------------------------------------ #


def _ok_cold(wc) -> RequestResult:
    """A clean cold (STORE) pass: full miss, two chunks stored."""
    return RequestResult(checksums=["a", "b"], total_chunks=2, hit_chunks=0)


def _ok_warm(wc) -> RequestResult:
    """A clean warm (RETRIEVE) pass: full hit, digests match the cold pass."""
    return RequestResult(checksums=["a", "b"], total_chunks=2, hit_chunks=2)


class _FakeRuntime:
    """A fake :class:`WorkerRuntime` recording orchestration for assertions.

    The runner is exercised through its public entry with these fakes, so the
    tests verify the concurrency contract (bands, barrier, verdict, teardown)
    without a real client or RPC. Behaviour is configured per row.
    """

    def __init__(
        self,
        *,
        cold=_ok_cold,
        warm=_ok_warm,
        setup_fail_workers=(),
        request_raise=None,
        unregister_ack=True,
        unregister_raise=False,
        event_log=None,
    ) -> None:
        self.setup_calls: list = []
        self.request_calls: list = []
        self.unregister_calls: list = []
        self.close_calls: list = []
        self.clients: dict = {}
        self.instance_ids: dict = {}
        self.bands: dict = {}
        self.setups_at_first_request: "int | None" = None
        self._cold = cold
        self._warm = warm
        self._setup_fail_workers = set(setup_fail_workers)
        self._request_raise = request_raise or {}
        self._unregister_ack = unregister_ack
        self._unregister_raise = unregister_raise
        # Optional thread-safe sink recording "setup:w" / "request:w" /
        # "close:w" so a test can assert the measured-window ordering.
        self._event_log = event_log

    def _log(self, tag: str) -> None:
        if self._event_log is not None:
            self._event_log(tag)

    def as_runtime(self) -> WorkerRuntime:
        return WorkerRuntime(
            setup=self.setup,
            process_request=self.process_request,
            unregister=self.unregister,
            close=self.close,
        )

    def setup(self, wc) -> None:
        self.setup_calls.append(wc.worker_id)
        self.instance_ids[wc.worker_id] = wc.instance_id
        self.bands[wc.worker_id] = (wc.band_base, wc.band_blocks)
        self._log("setup:%d" % wc.worker_id)
        # REGISTER is marked possibly-submitted before it can time out.
        wc.registered_maybe = True
        if wc.worker_id in self._setup_fail_workers:
            raise RuntimeError("REGISTER timed out for worker %d" % wc.worker_id)
        wc.client = object()
        self.clients[wc.worker_id] = wc.client

    def process_request(self, wc, seq_no, pass_label):
        if self.setups_at_first_request is None:
            self.setups_at_first_request = len(set(self.setup_calls))
        self.request_calls.append((wc.worker_id, seq_no, pass_label))
        self._log("request:%d" % wc.worker_id)
        exc = self._request_raise.get((wc.worker_id, pass_label))
        if exc is None:
            exc = self._request_raise.get(wc.worker_id)
        if exc is not None:
            raise exc
        maker = self._cold if pass_label == "cold" else self._warm
        return maker(wc)

    def unregister(self, wc) -> bool:
        self.unregister_calls.append(wc.worker_id)
        if self._unregister_raise:
            raise RuntimeError("UNREGISTER raised for worker %d" % wc.worker_id)
        return self._unregister_ack

    def close(self, wc) -> None:
        self.close_calls.append(wc.worker_id)
        self._log("close:%d" % wc.worker_id)


class TestConcurrentRunner:
    """The concurrency contract: private state, a gated measured phase, and a
    verdict that never reports a confident-but-wrong or unsafe run."""

    def test_n1_success(self) -> None:
        rt = _FakeRuntime()
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=1,
            end=3,
            total_blocks=1024,
            request_blocks=32,
        )
        assert res.verdict.valid
        assert res.verdict.server_reuse_safe
        assert not res.interrupted
        assert res.stats.total_requests == 3
        assert res.stats.checksum_ok == 3

    def test_n2_success_distinct_clients_and_instance_ids(self) -> None:
        rt = _FakeRuntime()
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=4,
            total_blocks=1024,
            request_blocks=32,
        )
        assert res.verdict.valid
        assert len(set(rt.instance_ids.values())) == 2
        assert len({id(c) for c in rt.clients.values()}) == 2
        assert res.stats.total_requests == 4

    def test_n2_non_overlapping_bands(self) -> None:
        rt = _FakeRuntime()
        run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=2,
            total_blocks=1000,
            request_blocks=32,
        )
        # 1000 // 2 = 500 blocks per band, adjacent and non-overlapping.
        assert rt.bands[0] == (0, 500)
        assert rt.bands[1] == (500, 500)

    def test_requests_start_only_after_all_setup(self) -> None:
        rt = _FakeRuntime()
        run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=2,
            total_blocks=1024,
            request_blocks=32,
        )
        # The start barrier gates the workload: both workers finished setup
        # before the first request was issued.
        assert rt.setups_at_first_request == 2

    def test_setup_failure_aborts_without_deadlock(self) -> None:
        rt = _FakeRuntime(setup_fail_workers=(0,))
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=2,
            total_blocks=1024,
            request_blocks=32,
        )
        # The run is invalid and no request was issued (the workload never
        # started); the test completing at all proves no deadlock.
        assert not res.verdict.valid
        assert rt.request_calls == []

    def test_one_worker_invalid_run_invalid_but_reuse_safe(self) -> None:
        def _bad_warm(wc):
            if wc.worker_id == 1:
                return RequestResult(total_chunks=2, failure="RETRIEVE retrieve_failed")
            return _ok_warm(wc)

        rt = _FakeRuntime(warm=_bad_warm)
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=2,
            total_blocks=1024,
            request_blocks=32,
        )
        assert not res.verdict.valid
        # A plain request failure invalidates the run but does not taint.
        assert res.verdict.server_reuse_safe

    def test_one_worker_tainted_run_tainted(self) -> None:
        def _tainted_cold(wc):
            if wc.worker_id == 0:
                return RequestResult(
                    total_chunks=2, failure="LOOKUP timeout", server_tainted=True
                )
            return _ok_cold(wc)

        rt = _FakeRuntime(cold=_tainted_cold)
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=2,
            total_blocks=1024,
            request_blocks=32,
        )
        assert not res.verdict.valid
        assert not res.verdict.server_reuse_safe

    def test_register_timeout_still_unregisters(self) -> None:
        rt = _FakeRuntime(setup_fail_workers=(0,))
        run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=1,
            end=1,
            total_blocks=1024,
            request_blocks=32,
        )
        # registered_maybe was set before REGISTER timed out, so teardown
        # still best-effort UNREGISTERs the (possibly created) context.
        assert 0 in rt.unregister_calls

    def test_unregister_timeout_makes_run_tainted(self) -> None:
        rt = _FakeRuntime(unregister_ack=False)
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=1,
            end=1,
            total_blocks=1024,
            request_blocks=32,
        )
        assert not res.verdict.server_reuse_safe

    def test_worker_exception_is_not_partial_valid(self) -> None:
        rt = _FakeRuntime(request_raise={0: RuntimeError("boom")})
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=1,
            end=1,
            total_blocks=1024,
            request_blocks=32,
        )
        assert not res.verdict.valid

    def test_ctrl_c_keeps_interrupt_semantics(self) -> None:
        rt = _FakeRuntime(request_raise={0: KeyboardInterrupt()})
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=1,
            end=1,
            total_blocks=1024,
            request_blocks=32,
        )
        assert res.interrupted
        assert not res.verdict.valid

    def test_checksum_mismatch_invalidates_run(self) -> None:
        def _bad_warm(wc):
            return RequestResult(checksums=["a", "X"], total_chunks=2, hit_chunks=2)

        rt = _FakeRuntime(warm=_bad_warm)
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=1,
            end=1,
            total_blocks=1024,
            request_blocks=32,
        )
        assert not res.verdict.valid
        assert res.stats.checksum_fail >= 1

    def test_band_too_small_is_rejected(self) -> None:
        rt = _FakeRuntime()
        with pytest.raises(ValueError, match="band"):
            run_concurrent_pairs(
                rt.as_runtime(),
                concurrency=4,
                end=4,
                total_blocks=100,
                request_blocks=64,
            )

    def test_sequence_numbers_are_globally_unique(self) -> None:
        # Round-robin partitioning of [start, end) gives every worker a
        # disjoint stride, so no two workers ever share a seq_no -- and
        # therefore no two requests share a request_id / cache-key namespace.
        rt = _FakeRuntime()
        run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=3,
            total_blocks=4096,
            request_blocks=32,
            start=0,
            end=12,
        )
        cold_seqs = [
            (worker_id, seq_no)
            for (worker_id, seq_no, pass_label) in rt.request_calls
            if pass_label == "cold"
        ]
        seq_values = [seq_no for (_wid, seq_no) in cold_seqs]
        # 3 workers round-robin over [0, 12): every seq distinct, all covered.
        assert len(seq_values) == 12
        assert set(seq_values) == set(range(12))
        # Worker w owns the stride start=w, step=concurrency.
        per_worker: dict[int, list[int]] = {}
        for worker_id, seq_no in cold_seqs:
            per_worker.setdefault(worker_id, []).append(seq_no)
        assert sorted(per_worker[0]) == [0, 3, 6, 9]
        assert sorted(per_worker[1]) == [1, 4, 7, 10]
        assert sorted(per_worker[2]) == [2, 5, 8, 11]

    def test_measurement_hooks_bracket_workload_only(self) -> None:
        # The measured window must open after all setup/REGISTER and close
        # after all workload but before any teardown/UNREGISTER.
        events: list = []
        lock = threading.Lock()

        def _log(tag: str) -> None:
            with lock:
                events.append(tag)

        rt = _FakeRuntime(event_log=_log)
        run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=2,
            total_blocks=1024,
            request_blocks=32,
            on_measure_start=lambda: _log("measure_start"),
            on_measure_stop=lambda: _log("measure_stop"),
        )
        start_idx = events.index("measure_start")
        stop_idx = events.index("measure_stop")
        setup_idx = [i for i, e in enumerate(events) if e.startswith("setup:")]
        request_idx = [i for i, e in enumerate(events) if e.startswith("request:")]
        teardown_idx = [i for i, e in enumerate(events) if e.startswith("close:")]
        # setup / REGISTER excluded: all before the window opens.
        assert max(setup_idx) < start_idx
        # workload included: every request inside the window.
        assert all(start_idx < i < stop_idx for i in request_idx)
        # teardown / UNREGISTER excluded: all after the window closes.
        assert stop_idx < min(teardown_idx)

    def test_main_thread_ctrl_c_while_orchestrator_waits(self) -> None:
        # Simulate a Ctrl-C landing on the orchestrator thread (here as it
        # opens the measured window, with every worker parked at the start
        # gate): barriers abort, teardown still runs, threads terminate, and
        # the run is reported interrupted -- never partial-valid.
        def _interrupt_on_start() -> None:
            raise KeyboardInterrupt

        rt = _FakeRuntime()
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            end=4,
            total_blocks=1024,
            request_blocks=32,
            on_measure_start=_interrupt_on_start,
        )
        assert res.interrupted
        assert not res.verdict.valid
        # Both worker threads terminated and tore down (no hang, no leak).
        assert set(rt.close_calls) == {0, 1}
        # The workload never started, so nothing partial was measured.
        assert rt.request_calls == []

    def test_seq_finite_range_round_robin(self) -> None:
        stop = threading.Event()
        w0 = list(_worker_seq_numbers(0, 2, start=0, end=5, stop_event=stop))
        w1 = list(_worker_seq_numbers(1, 2, start=0, end=5, stop_event=stop))
        assert w0 == [0, 2, 4]
        assert w1 == [1, 3]
        # Together the workers cover [0, 5) with no overlap.
        assert sorted(w0 + w1) == [0, 1, 2, 3, 4]

    def test_seq_infinite_range_is_unbounded(self) -> None:
        stop = threading.Event()
        gen = _worker_seq_numbers(0, 2, start=0, end=None, stop_event=stop)
        # end=None is the legacy infinite run: it yields forever, so take a
        # bounded prefix rather than draining it.
        assert list(itertools.islice(gen, 5)) == [0, 2, 4, 6, 8]
        # Setting the stop flag ends the generator on the next step.
        stop.set()
        assert list(gen) == []

    def test_seq_concurrency_one_is_contiguous(self) -> None:
        stop = threading.Event()
        seqs = list(_worker_seq_numbers(0, 1, start=0, end=4, stop_event=stop))
        assert seqs == [0, 1, 2, 3]

    def test_real_main_thread_interrupt_during_orchestrator_wait(
        self, monkeypatch
    ) -> None:
        # A *real* KeyboardInterrupt raised on the main (orchestrator) thread
        # via _thread.interrupt_main(), fired once a worker signals its
        # workload has started, while the orchestrator is blocked waiting for
        # the workers to finish. end=None keeps the workers running so the
        # orchestrator stays parked on the done barrier until the interrupt.
        #
        # interrupt_main() sets a pending exception delivered when the main
        # thread's blocking wait next returns, so we shorten the barrier
        # timeout to keep the test fast; the abort / teardown / interrupt path
        # it exercises is identical at the production timeout.
        monkeypatch.setattr(sv_runner, "_START_BARRIER_TIMEOUT_S", 1.0)
        started = threading.Event()

        def _cold_then_interrupt(wc):
            if not started.is_set():
                started.set()
                _thread.interrupt_main()
            return _ok_cold(wc)

        rt = _FakeRuntime(cold=_cold_then_interrupt)
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            total_blocks=1024,
            request_blocks=32,
            start=0,
            end=None,
        )
        assert res.interrupted
        assert not res.verdict.valid
        # Both barriers aborted, both worker threads tore down and terminated
        # (the test returning at all proves there was no deadlock).
        assert set(rt.close_calls) == {0, 1}

    def test_measurement_start_hook_error_invalidates_and_tears_down(self) -> None:
        def _boom() -> None:
            raise RuntimeError("profiler start failed")

        rt = _FakeRuntime()
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            total_blocks=1024,
            request_blocks=32,
            start=0,
            end=2,
            on_measure_start=_boom,
        )
        # A start-hook failure is invalid but not an interrupt; teardown still
        # runs on the worker threads, and the workload never opened.
        assert not res.verdict.valid
        assert not res.interrupted
        assert set(rt.close_calls) == {0, 1}
        assert rt.request_calls == []

    def test_measurement_stop_hook_error_invalidates_and_tears_down(self) -> None:
        def _boom() -> None:
            raise RuntimeError("profiler stop failed")

        rt = _FakeRuntime()
        res = run_concurrent_pairs(
            rt.as_runtime(),
            concurrency=2,
            total_blocks=1024,
            request_blocks=32,
            start=0,
            end=2,
            on_measure_stop=_boom,
        )
        # A stop-hook failure invalidates the run but teardown still completes
        # on the worker threads; the workload had already run.
        assert not res.verdict.valid
        assert not res.interrupted
        assert set(rt.close_calls) == {0, 1}
        assert len(rt.request_calls) > 0
