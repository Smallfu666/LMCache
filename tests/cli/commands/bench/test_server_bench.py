# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache bench server`` CLI command.

Covers:
- Sub-command registration under ``lmcache bench``
- Argument registration and defaults (including concurrency / workload)
- Pure helper functions (_build_token_ids, _make_key, _query_checksum)
- Concurrency partitioning (_band_blocks, _request_block_offset,
  _worker_seq_numbers)
- Stats aggregation (_merge_worker_stats, _bytes_per_token,
  _account_result) and the latency percentile summary
  (_add_latency_section)
- Workload config validation (_resolve_workload_config), including the
  --server-max-gpu-workers gate and the engine-driven narrow-scope gate
  (data path admitted only for --op pair --concurrency 1)
- Request-failure classification (LOOKUP / prefetch-poll / STORE /
  RETRIEVE timeouts and failures never pass silently)
- Workload purity contracts (RequestContract: store-only must not
  RETRIEVE, retrieve-only must not STORE)
- Session cleanup: END_SESSION always runs once a LOOKUP was accepted,
  and FREE_LOOKUP_LOCKS is issued on a contract violation that held locks
- The per-run nonce (fresh instance_id base + request_id prefix)
- Checksum fail-close (missing / short / mismatched digests invalidate)
- The driver / orchestrator abort path (invalid summary + exit 1, the
  retrieve-only pre-warm gate, teardown UNREGISTER failures, Ctrl-C)
- The two-phase start barrier (no request before the gate releases)
- Per-worker deterministic seeds (reproducible, distinct, global RNG
  untouched)
- The handle-mode checksum oracle (negative + positive control)
"""

# Standard
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from typing import cast
import argparse
import json
import threading

# Third Party
import msgspec
import pytest
import torch
import zmq

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.bench.server_bench import command as sv_cmd
from lmcache.cli.commands.bench.server_bench import helpers as sv_helpers
from lmcache.cli.commands.bench.server_bench.command import (
    WorkerStats,
    _account_result,
    _add_latency_section,
    _bytes_per_token,
    _drive_pair,
    _drive_store_pass,
    _expected_worker_requests,
    _merge_worker_stats,
    _prewarm_gate,
    _RequestFailureError,
    _resolve_workload_config,
    _run_phase,
    _teardown_worker_contexts,
    _worker_seq_numbers,
    _WorkerContext,
    run_server_bench,
)
from lmcache.cli.commands.bench.server_bench.helpers import (
    ChecksumMode,
    OpMode,
    RequestContract,
    RequestResult,
    _allocate_kv_cache,
    _band_blocks,
    _build_token_ids,
    _compute_client_checksums,
    _make_key,
    _poll_prefetch_status,
    _process_request,
    _query_checksum,
    _request_block_offset,
    _send_lookup,
    _send_unregister_kv_cache,
)
from lmcache.cli.metrics import Metrics
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
        assert sent["instance_id"] == 0
        assert sent["layerwise"] is False

    def test_instance_id_forwarded(self):
        base = "http://127.0.0.1:%d" % self.port
        _query_checksum(
            base,
            block_offset=0,
            num_blocks=2,
            block_size=2,
            chunk_size=2,
            instance_id=3,
        )
        assert self.server.received_payloads[-1]["instance_id"] == 3

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
#  Concurrency / workload argument registration                       #
# ------------------------------------------------------------------ #


class TestConcurrencyArguments:
    def test_defaults_preserve_legacy(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        """New flags default to the single-threaded pair behaviour."""
        args = parser.parse_args(["bench", "server"])
        assert args.concurrency == 1
        assert args.op == "pair"
        assert args.requests is None
        assert args.prefetch_poll_interval == 0.05
        assert args.checksum == "auto"

    def test_custom_values(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--concurrency",
                "4",
                "--op",
                "store-only",
                "--requests",
                "50",
                "--prefetch-poll-interval",
                "0.001",
                "--checksum",
                "off",
            ],
        )
        assert args.concurrency == 4
        assert args.op == "store-only"
        assert args.requests == 50
        assert args.prefetch_poll_interval == 0.001
        assert args.checksum == "off"

    def test_op_choices_rejected(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        with pytest.raises(SystemExit):
            parser.parse_args(["bench", "server", "--op", "bogus"])

    def test_server_max_gpu_workers_parsed(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        args = parser.parse_args(
            [
                "bench",
                "server",
                "--concurrency",
                "2",
                "--requests",
                "10",
                "--server-max-gpu-workers",
                "8",
            ],
        )
        assert args.server_max_gpu_workers == 8

    def test_old_server_max_workers_flag_removed(
        self,
        parser: argparse.ArgumentParser,
    ) -> None:
        # The flag was renamed to --server-max-gpu-workers.
        with pytest.raises(SystemExit):
            parser.parse_args(["bench", "server", "--server-max-workers", "8"])


# ------------------------------------------------------------------ #
#  _band_blocks (per-worker band size, concurrency correctness root)  #
# ------------------------------------------------------------------ #


class TestBandBlocks:
    """Each worker registers its own band-size KV cache. The band split
    must keep total memory flat and fail fast when it cannot hold a
    request; N=1 must return the full pool with no size check.
    """

    def test_single_worker_returns_full_pool(self) -> None:
        assert _band_blocks(1024, 1, 32) == 1024

    def test_single_worker_no_raise_when_request_larger(self) -> None:
        """N=1 keeps the legacy clamp: never raises even if a request is
        larger than the whole pool."""
        assert _band_blocks(1024, 1, 2048) == 1024

    def test_even_split(self) -> None:
        assert _band_blocks(1024, 4, 32) == 256

    def test_floor_division(self) -> None:
        # Total need not divide evenly; the band floors.
        assert _band_blocks(1000, 3, 10) == 333

    def test_band_too_small_raises(self) -> None:
        """Too many workers for the block pool must fail loudly."""
        with pytest.raises(ValueError, match="block partition too small"):
            # 4 workers over 64 blocks -> 16-block band < 32-block request.
            _band_blocks(64, 4, 32)

    def test_bad_num_workers_raises(self) -> None:
        with pytest.raises(ValueError, match="num_workers"):
            _band_blocks(1024, 0, 8)


# ------------------------------------------------------------------ #
#  _request_block_offset (per-request slice within a worker's band)   #
# ------------------------------------------------------------------ #


class TestRequestBlockOffset:
    """The per-request offset is local to a worker's own band-size KV
    cache, so it must (a) match the historical single-worker formula when
    the band is the full pool, and (b) always stay inside the band.
    """

    @staticmethod
    def _legacy(seq_no: int, num_blocks: int, total_blocks: int) -> int:
        usable = max(total_blocks - num_blocks, 1)
        return (seq_no * num_blocks) % usable

    @pytest.mark.parametrize("seq_no", [0, 1, 5, 17, 128, 1000])
    def test_full_band_matches_legacy(self, seq_no: int) -> None:
        # band == total_blocks reproduces the legacy single-worker layout.
        assert _request_block_offset(seq_no, 32, 1024) == self._legacy(seq_no, 32, 1024)

    def test_clamp_when_request_larger_than_band(self) -> None:
        # usable clamps to 1 -> offset 0 (never addresses out of range).
        assert _request_block_offset(7, 2048, 1024) == 0

    def test_offset_stays_inside_band(self) -> None:
        band = 256
        num_blocks = 32
        for seq_no in range(0, 500):
            off = _request_block_offset(seq_no, num_blocks, band)
            assert 0 <= off
            assert off + num_blocks <= band


# ------------------------------------------------------------------ #
#  _worker_seq_numbers (sequence-space partition)                     #
# ------------------------------------------------------------------ #


class TestWorkerSeqNumbers:
    def test_single_worker_matches_range(self) -> None:
        seqs = list(_worker_seq_numbers(0, 0, 1, None, 10, None))
        assert seqs == list(range(0, 10))

    def test_single_worker_honours_start(self) -> None:
        seqs = list(_worker_seq_numbers(5, 0, 1, None, 10, None))
        assert seqs == [5, 6, 7, 8, 9]

    def test_stride_partition_is_disjoint_and_covers(self) -> None:
        num_workers = 3
        end = 20
        all_seqs: list[int] = []
        per_worker = []
        for w in range(num_workers):
            s = list(_worker_seq_numbers(0, w, num_workers, None, end, None))
            per_worker.append(set(s))
            all_seqs.extend(s)
        # Union covers [0, end) exactly, with no duplicates.
        assert sorted(all_seqs) == list(range(0, end))
        for a in range(num_workers):
            for b in range(a + 1, num_workers):
                assert per_worker[a].isdisjoint(per_worker[b])

    def test_requests_cap(self) -> None:
        # 3 requests per worker, stride 2, start 0, worker 1.
        seqs = list(_worker_seq_numbers(0, 1, 2, 3, None, None))
        assert seqs == [1, 3, 5]

    def test_stop_event_halts(self) -> None:
        stop = threading.Event()
        gen = _worker_seq_numbers(0, 0, 1, None, None, stop)
        assert next(gen) == 0
        assert next(gen) == 1
        stop.set()
        with pytest.raises(StopIteration):
            next(gen)


# ------------------------------------------------------------------ #
#  _bytes_per_token                                                   #
# ------------------------------------------------------------------ #


class TestBytesPerToken:
    def test_single_group(self) -> None:
        # kv_size=2, nh=8, hs=128, float16 (2 bytes), 32 layers.
        group = SimpleNamespace(
            shape_desc=SimpleNamespace(kv_size=2, nh=8, hs=128, nl=32),
            dtype=torch.float16,
        )
        # 2 * 8 * 128 * 2 * 32 = 131072
        assert _bytes_per_token([group]) == 131072

    def test_multi_group_sums(self) -> None:
        g_a = SimpleNamespace(
            shape_desc=SimpleNamespace(kv_size=1, nh=1, hs=128, nl=4),
            dtype=torch.float16,
        )
        g_b = SimpleNamespace(
            shape_desc=SimpleNamespace(kv_size=2, nh=8, hs=128, nl=28),
            dtype=torch.float16,
        )
        # g_a: 1*1*128*2*4 = 1024 ; g_b: 2*8*128*2*28 = 114688
        assert _bytes_per_token([g_a, g_b]) == 1024 + 114688

    def test_empty(self) -> None:
        assert _bytes_per_token([]) == 0


# ------------------------------------------------------------------ #
#  _merge_worker_stats                                                #
# ------------------------------------------------------------------ #


class TestMergeWorkerStats:
    def test_merge_concatenates_and_sums(self) -> None:
        s0 = WorkerStats(worker_id=0)
        s0.cold_lookup_ms.extend([1.0, 2.0])
        s0.warm_retrieve_ms.append(3.0)
        s0.total_requests = 2
        s0.checksum_ok = 2
        s0.store_tokens = 100
        s0.retrieve_tokens = 40

        s0.store_attempted = 2
        s0.store_ok = 2
        s0.retrieve_attempted = 1
        s0.retrieve_timeout = 1

        s1 = WorkerStats(worker_id=1)
        s1.cold_lookup_ms.append(5.0)
        s1.total_requests = 3
        s1.checksum_fail = 1
        s1.store_tokens = 50
        s1.retrieve_tokens = 60
        s1.store_attempted = 3
        s1.store_ok = 2
        s1.store_failed = 1
        s1.retrieve_attempted = 3
        s1.retrieve_ok = 3

        merged = _merge_worker_stats([s0, s1])
        assert merged.worker_id == -1
        assert sorted(merged.cold_lookup_ms) == [1.0, 2.0, 5.0]
        assert merged.warm_retrieve_ms == [3.0]
        assert merged.total_requests == 5
        assert merged.checksum_ok == 2
        assert merged.checksum_fail == 1
        assert merged.store_tokens == 150
        assert merged.retrieve_tokens == 100
        assert merged.store_attempted == 5
        assert merged.store_ok == 4
        assert merged.store_failed == 1
        assert merged.retrieve_attempted == 4
        assert merged.retrieve_ok == 3
        assert merged.retrieve_timeout == 1

    def test_merge_carries_first_error(self) -> None:
        s0 = WorkerStats(worker_id=0)
        s1 = WorkerStats(worker_id=1, error="RuntimeError: boom")
        merged = _merge_worker_stats([s0, s1])
        assert merged.error == "RuntimeError: boom"


# ------------------------------------------------------------------ #
#  _account_result (failure / timeout accounting, tokens on success)  #
# ------------------------------------------------------------------ #


class TestAccountResult:
    """Each attempted STORE / RETRIEVE must land in exactly one outcome
    bucket, and an unattempted op (``None`` status) must not be counted.
    """

    def test_store_success(self) -> None:
        s = WorkerStats(worker_id=0)
        _account_result(s, RequestResult(store_status="stored"))
        assert (s.store_attempted, s.store_ok, s.store_failed, s.store_timeout) == (
            1,
            1,
            0,
            0,
        )

    def test_store_failed(self) -> None:
        s = WorkerStats(worker_id=0)
        _account_result(s, RequestResult(store_status="store_failed"))
        assert (s.store_attempted, s.store_ok, s.store_failed, s.store_timeout) == (
            1,
            0,
            1,
            0,
        )

    def test_store_timeout(self) -> None:
        s = WorkerStats(worker_id=0)
        _account_result(s, RequestResult(store_status="timeout"))
        assert (s.store_attempted, s.store_ok, s.store_failed, s.store_timeout) == (
            1,
            0,
            0,
            1,
        )

    def test_retrieve_outcomes(self) -> None:
        s = WorkerStats(worker_id=0)
        _account_result(s, RequestResult(retrieve_status="retrieved"))
        _account_result(s, RequestResult(retrieve_status="retrieve_failed"))
        _account_result(s, RequestResult(retrieve_status="timeout"))
        assert s.retrieve_attempted == 3
        assert s.retrieve_ok == 1
        assert s.retrieve_failed == 1
        assert s.retrieve_timeout == 1

    def test_none_status_not_counted(self) -> None:
        s = WorkerStats(worker_id=0)
        _account_result(s, RequestResult())  # both statuses None
        assert s.store_attempted == 0
        assert s.retrieve_attempted == 0

    def test_pair_result_counts_both(self) -> None:
        s = WorkerStats(worker_id=0)
        _account_result(
            s, RequestResult(store_status="stored", retrieve_status="retrieved")
        )
        assert s.store_ok == 1
        assert s.retrieve_ok == 1


# ------------------------------------------------------------------ #
#  _add_latency_section (p95 added)                                   #
# ------------------------------------------------------------------ #


class TestLatencySection:
    def test_reports_p95_and_percentiles(self) -> None:
        metrics = Metrics(title="t")
        _add_latency_section(
            metrics, "warm_retrieve", "Warm Retrieve (ms)", list(range(1, 101))
        )
        d = metrics.to_dict()["metrics"]["warm_retrieve"]
        assert d["warm_retrieve_count"] == 100
        assert d["warm_retrieve_min"] == 1
        assert d["warm_retrieve_max"] == 100
        # ceil(100*0.95)-1 = 94 -> sorted[94] = 95
        assert d["warm_retrieve_p95"] == 95
        assert d["warm_retrieve_p99"] == 99

    def test_empty_list_adds_nothing(self) -> None:
        metrics = Metrics(title="t")
        _add_latency_section(metrics, "cold_store", "Cold Store (ms)", [])
        assert "cold_store" not in metrics.to_dict()["metrics"]


# ------------------------------------------------------------------ #
#  _resolve_workload_config (validation + auto checksum)              #
# ------------------------------------------------------------------ #


def _workload_args(**overrides) -> argparse.Namespace:
    """Build a Namespace with workload-config defaults, overridable.

    ``mode``/``transfer_mode`` default to the handle path (gpu + auto), so
    the engine-driven narrow-scope gate is not tripped unless a test opts
    into the data path explicitly.
    """
    base = {
        "op": "pair",
        "concurrency": 1,
        "requests": None,
        "end": None,
        "checksum": "auto",
        "prefetch_poll_interval": 0.05,
        "mode": "gpu",
        "transfer_mode": "auto",
        "server_max_gpu_workers": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestResolveWorkloadConfig:
    def test_defaults_pair_checksum_on(self) -> None:
        cfg = _resolve_workload_config(_workload_args())
        assert cfg.op is OpMode.PAIR
        assert cfg.num_workers == 1
        assert cfg.checksum is ChecksumMode.ON

    def test_auto_checksum_off_for_store_only(self) -> None:
        cfg = _resolve_workload_config(_workload_args(op="store-only", requests=5))
        assert cfg.checksum is ChecksumMode.OFF

    def test_explicit_checksum_on_ok_for_pair(self) -> None:
        cfg = _resolve_workload_config(_workload_args(op="pair", checksum="on"))
        assert cfg.checksum is ChecksumMode.ON

    def test_checksum_on_rejected_for_store_only(self) -> None:
        with pytest.raises(ValueError, match="checksum on"):
            _resolve_workload_config(
                _workload_args(op="store-only", requests=5, checksum="on")
            )

    def test_checksum_on_rejected_for_retrieve_only(self) -> None:
        with pytest.raises(ValueError, match="checksum on"):
            _resolve_workload_config(
                _workload_args(op="retrieve-only", end=10, checksum="on")
            )

    def test_checksum_off_allowed_for_store_only(self) -> None:
        cfg = _resolve_workload_config(
            _workload_args(op="store-only", requests=5, checksum="off")
        )
        assert cfg.checksum is ChecksumMode.OFF

    def test_poll_interval_non_finite_rejected(self) -> None:
        with pytest.raises(ValueError, match="prefetch-poll-interval"):
            _resolve_workload_config(
                _workload_args(prefetch_poll_interval=float("inf"))
            )

    def test_poll_interval_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="prefetch-poll-interval"):
            _resolve_workload_config(
                _workload_args(prefetch_poll_interval=float("nan"))
            )

    def test_poll_interval_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="prefetch-poll-interval"):
            _resolve_workload_config(_workload_args(prefetch_poll_interval=-1.0))

    def test_requests_and_end_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            _resolve_workload_config(_workload_args(requests=5, end=10))

    def test_concurrency_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="concurrency"):
            _resolve_workload_config(_workload_args(concurrency=0))

    def test_concurrent_forever_rejected(self) -> None:
        with pytest.raises(ValueError, match="forever"):
            _resolve_workload_config(_workload_args(concurrency=2))

    def test_concurrent_bounded_ok(self) -> None:
        cfg = _resolve_workload_config(
            _workload_args(concurrency=2, requests=10, server_max_gpu_workers=4)
        )
        assert cfg.num_workers == 2
        assert cfg.requests_per_worker == 10
        assert cfg.server_max_gpu_workers == 4

    def test_concurrency_gt1_requires_server_max_gpu_workers(self) -> None:
        with pytest.raises(ValueError, match="server-max-gpu-workers"):
            _resolve_workload_config(_workload_args(concurrency=2, requests=10))

    def test_concurrency_exceeding_server_max_gpu_workers_refused(self) -> None:
        with pytest.raises(ValueError, match="exceeds --server-max-gpu-workers"):
            _resolve_workload_config(
                _workload_args(concurrency=8, requests=10, server_max_gpu_workers=4)
            )

    def test_server_max_gpu_workers_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="server-max-gpu-workers"):
            _resolve_workload_config(_workload_args(server_max_gpu_workers=0))

    # -- use_handle resolution + engine-driven narrow-scope gate --------

    def test_use_handle_gpu_auto(self) -> None:
        cfg = _resolve_workload_config(_workload_args(mode="gpu", transfer_mode="auto"))
        assert cfg.use_handle is True

    def test_use_handle_cpu_auto_is_data_path(self) -> None:
        # cpu + auto -> engine_driven (data path); allowed for pair/N=1.
        cfg = _resolve_workload_config(_workload_args(mode="cpu", transfer_mode="auto"))
        assert cfg.use_handle is False

    def test_use_handle_cpu_lmcache_driven(self) -> None:
        cfg = _resolve_workload_config(
            _workload_args(mode="cpu", transfer_mode="lmcache_driven")
        )
        assert cfg.use_handle is True

    def test_engine_driven_pair_n1_allowed(self) -> None:
        # The historical data-path sanity test stays admissible (default
        # checksum auto -> on in pair mode satisfies the new guard).
        cfg = _resolve_workload_config(
            _workload_args(mode="cpu", transfer_mode="engine_driven", op="pair")
        )
        assert cfg.use_handle is False
        assert cfg.op is OpMode.PAIR
        assert cfg.checksum is ChecksumMode.ON

    def test_engine_driven_checksum_off_rejected(self) -> None:
        # The data path has no oracle without checksums, so --checksum off
        # must be refused even for the admissible N=1 pair sanity test.
        with pytest.raises(ValueError, match="requires --checksum on"):
            _resolve_workload_config(
                _workload_args(
                    mode="cpu",
                    transfer_mode="engine_driven",
                    op="pair",
                    checksum="off",
                )
            )

    def test_engine_driven_gpu_rejected(self) -> None:
        # GPU + engine_driven is not a validated path.
        with pytest.raises(ValueError, match="only supported on"):
            _resolve_workload_config(
                _workload_args(
                    mode="gpu",
                    transfer_mode="engine_driven",
                    op="pair",
                )
            )

    def test_engine_driven_store_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="engine-driven data path"):
            _resolve_workload_config(
                _workload_args(
                    mode="cpu",
                    transfer_mode="engine_driven",
                    op="store-only",
                    requests=5,
                    checksum="off",
                )
            )

    def test_engine_driven_retrieve_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="engine-driven data path"):
            _resolve_workload_config(
                _workload_args(
                    mode="cpu",
                    transfer_mode="engine_driven",
                    op="retrieve-only",
                    end=10,
                    checksum="off",
                )
            )

    def test_engine_driven_concurrency_rejected(self) -> None:
        # Even pair mode is refused on the data path once N > 1.
        with pytest.raises(ValueError, match="engine-driven data path"):
            _resolve_workload_config(
                _workload_args(
                    mode="cpu",
                    transfer_mode="engine_driven",
                    op="pair",
                    concurrency=2,
                    requests=10,
                    server_max_gpu_workers=4,
                )
            )

    def test_cpu_auto_concurrency_rejected(self) -> None:
        # cpu + auto is the data path too, so N > 1 is refused.
        with pytest.raises(ValueError, match="engine-driven data path"):
            _resolve_workload_config(
                _workload_args(
                    mode="cpu",
                    transfer_mode="auto",
                    op="pair",
                    concurrency=2,
                    requests=10,
                    server_max_gpu_workers=4,
                )
            )

    def test_gpu_concurrency_ok_handle_path(self) -> None:
        # The handle path (gpu) admits concurrent throughput runs.
        cfg = _resolve_workload_config(
            _workload_args(
                mode="gpu",
                op="store-only",
                concurrency=4,
                requests=10,
                server_max_gpu_workers=8,
                checksum="off",
            )
        )
        assert cfg.use_handle is True
        assert cfg.num_workers == 4

    def test_retrieve_only_requires_bound(self) -> None:
        with pytest.raises(ValueError, match="retrieve-only requires"):
            _resolve_workload_config(_workload_args(op="retrieve-only"))

    def test_retrieve_only_bounded_ok(self) -> None:
        cfg = _resolve_workload_config(_workload_args(op="retrieve-only", end=10))
        assert cfg.op is OpMode.RETRIEVE_ONLY
        assert cfg.checksum is ChecksumMode.OFF


# ------------------------------------------------------------------ #
#  Fake RPC layer (client-level timeout / outcome injection)          #
# ------------------------------------------------------------------ #

# These tests patch ``sv_helpers._call`` — the single choke point every
# protocol operation goes through — so the *real* ``_process_request``,
# driver, and orchestrator code paths run against injected client-level
# outcomes (RPC timeout, store rejection, ...).


# Stand-in client for paths where the RPC layer itself is patched out;
# nothing ever calls methods on it.
_DUMMY_CLIENT = cast(MessageQueueClient, object())


def _dispatching_call(behavior, calls):
    """Build a ``_call`` replacement dispatching on request type.

    ``behavior`` maps RequestType -> value | callable(payloads) -> value.
    The sentinel ``sv_helpers._TIMEOUT`` simulates an RPC timeout.
    Unlisted request types reply void (``None``). Every call is recorded
    into ``calls`` for issued-operation assertions.
    """

    def _fake_call(client, request_type, payloads, timeout_s=10.0):
        calls.append(request_type)
        action = behavior.get(request_type)
        if callable(action):
            return action(payloads)
        return action

    return _fake_call


def _request_kwargs(**overrides):
    """Baseline kwargs for driving _process_request in handle mode."""
    base = dict(
        num_tokens=511,  # +1 seq token = 512 = 2 chunks of 256
        chunk_size=256,
        http_base="",
        block_size=16,
        total_blocks=1024,
        num_engine_group_infos=1,
        use_gpu=False,
        use_handle=True,
        client_tensors=None,
        server_pool=None,
        worker_id=0,
        instance_id=0,
        poll_interval=0.0,
        checksum=ChecksumMode.OFF,
    )
    base.update(overrides)
    return base


class TestRequestFailureClassification:
    """B1: LOOKUP / prefetch-poll / STORE / RETRIEVE failures and
    timeouts must surface as request-level failures, never as a skip, a
    miss, or a silently absorbed status."""

    def test_lookup_timeout_is_failure_not_none(self, monkeypatch) -> None:
        calls: list = []
        monkeypatch.setattr(
            sv_helpers,
            "_call",
            _dispatching_call({RequestType.LOOKUP: sv_helpers._TIMEOUT}, calls),
        )
        result = _process_request(
            _DUMMY_CLIENT, 0, pass_label="cold", **_request_kwargs()
        )
        assert result is not None
        assert "LOOKUP timeout" in result.failure
        # The LOOKUP was submitted, so its server effect is unknown: the
        # server may hold session / prefetch state -> taint (restart needed).
        assert result.server_tainted is True
        # Nothing was stored / retrieved on top of the timeout.
        assert result.store_status is None
        assert result.retrieve_status is None
        assert RequestType.STORE not in calls
        assert RequestType.RETRIEVE not in calls

    def test_prefetch_poll_timeout_is_failure_not_miss(self, monkeypatch) -> None:
        """A poll RPC timeout must not be converted into hit_chunks=0
        (which would silently issue a STORE as if it were a miss)."""
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,  # void = success
            RequestType.QUERY_PREFETCH_STATUS: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None
        assert "prefetch status poll failed" in result.failure
        assert result.store_status is None
        assert RequestType.STORE not in calls

    def test_store_timeout_marks_failure(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss
            RequestType.STORE: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None
        assert result.store_status == "timeout"
        assert "STORE timeout" in result.failure
        # Diagnostics retained: the attempt is still accounted.
        stats = WorkerStats(worker_id=0)
        _account_result(stats, result)
        assert stats.store_timeout == 1

    def test_retrieve_failure_marks_failure(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 2,  # full hit
            RequestType.RETRIEVE: (0, False),  # server declined
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="warm",
            contract=RequestContract.RETRIEVE_HIT,
            **_request_kwargs(),
        )
        assert result is not None
        assert result.retrieve_status == "retrieve_failed"
        assert "RETRIEVE retrieve_failed" in result.failure


class TestWorkloadPurityContract:
    """B2: the throughput passes declare an explicit per-request
    contract; the operation that would break it is never issued."""

    def test_store_only_hit_is_violation_and_no_retrieve(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 2,  # unexpected full hit
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None
        assert "contract violation" in result.failure
        assert "full miss" in result.failure
        assert result.retrieve_status is None
        assert result.store_status is None
        assert RequestType.RETRIEVE not in calls
        assert RequestType.STORE not in calls

    def test_retrieve_only_miss_is_violation_and_no_store(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 1,  # partial hit of 2
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="warm",
            contract=RequestContract.RETRIEVE_HIT,
            **_request_kwargs(),
        )
        assert result is not None
        assert "contract violation" in result.failure
        assert "full hit" in result.failure
        assert result.store_status is None
        assert result.retrieve_status is None
        assert RequestType.STORE not in calls
        assert RequestType.RETRIEVE not in calls

    def test_pair_contract_allows_partial_hit(self, monkeypatch) -> None:
        """PAIR keeps the historical mixed flow: retrieve the hit
        portion, store the miss portion, no violation."""
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 1,  # 1 of 2 chunks hit
            RequestType.RETRIEVE: (0, True),
            RequestType.STORE: (0, True),
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.PAIR,
            **_request_kwargs(),
        )
        assert result is not None
        assert result.failure == ""
        assert result.retrieve_status == "retrieved"
        assert result.store_status == "stored"


class TestDriverFailureAbort:
    """B1: a request failure aborts the worker through _run_phase and
    surfaces as a worker error (-> invalid run)."""

    @staticmethod
    def _worker_context(kwargs):
        return _WorkerContext(
            worker_id=0,
            instance_id=0,
            client=_DUMMY_CLIENT,  # only the patched RPC layer touches it
            band_blocks=1024,
            request_kwargs=kwargs,
        )

    def test_store_timeout_aborts_run_phase(self, monkeypatch) -> None:
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,
            RequestType.STORE: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, []))
        wc = self._worker_context(_request_kwargs())
        stats = [WorkerStats(worker_id=0)]
        stop = threading.Event()
        _run_phase(
            [wc],
            _drive_store_pass,
            stats_list=stats,
            num_workers=1,
            start=0,
            requests_per_worker=5,
            end=None,
            interval=0.0,
            stop_event=stop,
            progress=None,
            inline=False,
        )
        assert stats[0].error
        assert "STORE timeout" in stats[0].error
        assert stop.is_set()
        # Aborted on the first request; no blind continuation.
        assert stats[0].total_requests == 1
        merged = _merge_worker_stats(stats)
        assert merged.error

    def test_run_phase_records_taint_from_server_tainted_error(self) -> None:
        # A ServerTaintedError escaping the driver (body raised + END_SESSION
        # timeout) must set stats.server_tainted so the run reports Server
        # reuse safe: no, not just a generic error.
        def _drive_tainted(wc, stats, **kwargs):
            raise sv_helpers.ServerTaintedError("cleanup timed out")

        wc = self._worker_context(_request_kwargs())
        stats = [WorkerStats(worker_id=0)]
        _run_phase(
            [wc],
            _drive_tainted,
            stats_list=stats,
            num_workers=1,
            start=0,
            requests_per_worker=5,
            end=None,
            interval=0.0,
            stop_event=threading.Event(),
            progress=None,
            inline=False,
        )
        assert stats[0].error
        assert stats[0].server_tainted is True
        assert _merge_worker_stats(stats).server_tainted is True

    def test_on_all_ready_failure_releases_gate_and_marks_phase_error(
        self, monkeypatch
    ) -> None:
        """Reliability: if the measured-phase start hook raises, _run_phase
        still releases the start gate (no non-daemon-thread hang), sets
        stop_event so released workers issue nothing, and returns a
        non-empty phase-error string so the run is marked invalid."""
        calls: list = []
        behavior = {RequestType.LOOKUP: None, RequestType.QUERY_PREFETCH_STATUS: 0}
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        wc = self._worker_context(_request_kwargs())
        stats = [WorkerStats(worker_id=0)]
        stop = threading.Event()

        def _boom() -> None:
            raise RuntimeError("profiler start failed")

        phase_error = _run_phase(
            [wc],
            _drive_store_pass,
            stats_list=stats,
            num_workers=1,
            start=0,
            requests_per_worker=5,
            end=None,
            interval=0.0,
            stop_event=stop,
            progress=None,
            inline=False,
            on_all_ready=_boom,
        )
        assert phase_error
        assert "profiler start failed" in phase_error
        assert stop.is_set()
        # Gate released only after stop_event was set, so the worker
        # observed the stop before issuing any request.
        assert stats[0].total_requests == 0
        assert calls == []

    def test_pair_zero_transfer_is_invalid(self, monkeypatch) -> None:
        """A pair request that stores/verifies nothing (fewer tokens than
        one chunk -> _process_request returns None) invalidates the run
        instead of being counted as a valid, fully-verified request."""
        calls: list = []
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call({}, calls))
        # num_tokens=0 -> a single seq token -> num_full_tokens == 0 ->
        # _process_request returns None before any RPC is issued.
        wc = self._worker_context(_request_kwargs(num_tokens=0, chunk_size=256))
        stats = [WorkerStats(worker_id=0)]
        stop = threading.Event()
        _run_phase(
            [wc],
            _drive_pair,
            stats_list=stats,
            num_workers=1,
            start=0,
            requests_per_worker=3,
            end=None,
            interval=0.0,
            stop_event=stop,
            progress=None,
            inline=False,
        )
        assert stats[0].error
        assert "nothing" in stats[0].error
        assert stats[0].total_requests == 0
        assert calls == []

    def test_prewarm_store_failure_blocks_phase_b(self) -> None:
        """Codex #3 at the gate level: a short / failed pre-warm must
        return a non-empty reason so Phase B never starts."""
        s = WorkerStats(worker_id=0)
        s.error = "_RequestFailureError: [w0 seq 0 store] STORE store_failed"
        assert _prewarm_gate(
            [s], start=0, num_workers=1, requests_per_worker=5, end=None
        )

        # No error, but completed short of the expected count.
        s2 = WorkerStats(worker_id=0)
        s2.total_requests = 3
        s2.store_ok = 3
        msg = _prewarm_gate(
            [s2], start=0, num_workers=1, requests_per_worker=5, end=None
        )
        assert "3/5" in msg

        # Completed, but one store did not succeed.
        s3 = WorkerStats(worker_id=0)
        s3.total_requests = 5
        s3.store_ok = 4
        msg = _prewarm_gate(
            [s3], start=0, num_workers=1, requests_per_worker=5, end=None
        )
        assert "stored 4/5" in msg

        # Clean pre-warm passes the gate.
        s4 = WorkerStats(worker_id=0)
        s4.total_requests = 5
        s4.store_ok = 5
        assert (
            _prewarm_gate([s4], start=0, num_workers=1, requests_per_worker=5, end=None)
            == ""
        )

    def test_prewarm_gate_rejects_server_tainted(self) -> None:
        # A pre-warm that stored everything but tainted the server (e.g. an
        # END_SESSION timeout) must not start Phase B: Phase B uses fresh
        # stats, so the taint would otherwise be lost and the run would
        # wrongly report Server reuse safe: yes.
        s = WorkerStats(worker_id=0)
        s.total_requests = 5
        s.store_ok = 5
        s.server_tainted = True
        msg = _prewarm_gate(
            [s], start=0, num_workers=1, requests_per_worker=5, end=None
        )
        assert "tainted" in msg


class TestExpectedWorkerRequests:
    def test_requests_mode(self) -> None:
        assert _expected_worker_requests(0, 2, 4, 7, None) == 7

    def test_end_mode_even(self) -> None:
        # [0, 20) over 4 workers -> 5 each.
        for w in range(4):
            assert _expected_worker_requests(0, w, 4, None, 20) == 5

    def test_end_mode_uneven_matches_generator(self) -> None:
        for w in range(3):
            expected = _expected_worker_requests(5, w, 3, None, 21)
            actual = len(list(_worker_seq_numbers(5, w, 3, None, 21, None)))
            assert expected == actual

    def test_worker_beyond_end(self) -> None:
        assert _expected_worker_requests(10, 2, 4, None, 11) == 0

    def test_unbounded_raises(self) -> None:
        with pytest.raises(ValueError, match="unbounded"):
            _expected_worker_requests(0, 0, 1, None, None)


# ------------------------------------------------------------------ #
#  Orchestrator-level invalid + exit path (B1)                        #
# ------------------------------------------------------------------ #


class _StubMetricsCommand:
    """Minimal BaseCommand stand-in: create_metrics returns a real
    Metrics so _emit_server_bench_metrics runs its actual code."""

    def create_metrics(self, title, args, width=64):
        return Metrics(title=title)


def _stub_command() -> BaseCommand:
    """The stub typed as BaseCommand (only create_metrics is used)."""
    return cast(BaseCommand, _StubMetricsCommand())


class _CapturingMetricsCommand:
    """Like _StubMetricsCommand but keeps the emitted Metrics for asserts."""

    def __init__(self) -> None:
        self.metrics: Metrics | None = None

    def create_metrics(self, title, args, width=64):
        self.metrics = Metrics(title=title)
        return self.metrics


class _FakeMQClient:
    """MessageQueueClient stand-in for orchestrator tests (the request
    path itself is stubbed at _process_request level)."""

    def __init__(self, url, ctx) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _orchestrator_args(**overrides) -> argparse.Namespace:
    base = dict(
        rpc_url="tcp://127.0.0.1:1",
        mode="cpu",
        # Handle path (lmcache_driven) so the throughput / concurrency
        # orchestrator scenarios are not refused by the engine-driven
        # narrow-scope gate; the request path itself is stubbed anyway.
        transfer_mode="lmcache_driven",
        num_tokens=511,
        kvcache_shape_spec="(2,64,16,2,4):float16:2",
        num_blocks=64,
        block_size=16,
        start=0,
        end=None,
        interval=0.0,
        url="http://127.0.0.1:1",
        concurrency=1,
        op="store-only",
        requests=2,
        prefetch_poll_interval=0.0,
        checksum="auto",
        flamegraph="off",
        profile_server_pid=0,
        flamegraph_mode="gil",
        flamegraph_output="",
        flamegraph_scripts_dir="",
        quiet=True,
        server_max_gpu_workers=None,
        server_commit="",
        server_image="",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def orchestrator_env(monkeypatch):
    """Patch the orchestrator's collaborators so run_server_bench runs
    end-to-end (config resolution, phase scheduling, validity
    resolution, metrics emission, exit code) without a live server."""
    # First Party
    import lmcache.v1.multiprocess.mq as mq_mod

    monkeypatch.setattr(mq_mod, "MessageQueueClient", _FakeMQClient)
    monkeypatch.setattr(sv_cmd, "_get_chunk_size", lambda client: 256)

    def _fake_setup(wc, **kwargs):
        wc.request_kwargs = {}
        wc.registered = False  # keeps real teardown to a client close

    monkeypatch.setattr(sv_cmd, "_setup_worker_context", _fake_setup)
    return monkeypatch


class TestOrchestratorInvalidExit:
    """B1 end-to-end: a client-level timeout surfaces as an invalid run
    and a non-zero exit through the real orchestrator."""

    def test_store_timeout_run_exits_1(self, orchestrator_env) -> None:
        pass_labels: list[str] = []

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            pass_labels.append(pass_label)
            return RequestResult(
                store_status="timeout",
                total_chunks=2,
                failure="STORE timeout (seq %d, cold pass)" % seq_no,
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(_stub_command(), _orchestrator_args())
        assert excinfo.value.code == 1
        # Aborted on the first failing request.
        assert pass_labels == ["cold"]

    def test_clean_store_only_run_exits_0(self, orchestrator_env) -> None:
        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            return RequestResult(
                store_status="stored",
                store_tokens=512,
                total_chunks=2,
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        # No SystemExit: the run is valid and returns normally.
        run_server_bench(_stub_command(), _orchestrator_args())

    def test_unregister_timeout_reports_server_unsafe(self, orchestrator_env) -> None:
        # A teardown UNREGISTER that times out leaves the context possibly
        # held on the server: the run must be invalid (exit 1) AND report
        # Server reuse safe: no, not just a non-zero exit.
        def _fake_setup(wc, **kwargs):
            wc.request_kwargs = {}
            wc.registered = True  # force teardown to attempt UNREGISTER

        orchestrator_env.setattr(sv_cmd, "_setup_worker_context", _fake_setup)
        orchestrator_env.setattr(
            sv_cmd,
            "_send_unregister_kv_cache_batch",
            lambda clients_and_ids, use_handle: [False] * len(clients_and_ids),
        )

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            return RequestResult(
                store_status="stored", store_tokens=512, total_chunks=2
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        cmd = _CapturingMetricsCommand()
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(cast(BaseCommand, cmd), _orchestrator_args())
        assert excinfo.value.code == 1
        assert cmd.metrics is not None
        d = cmd.metrics.to_dict()["metrics"]
        assert d["results"]["valid"] == "no"
        assert d["results"]["server_reuse_safe"] == "no"

    def test_prewarm_failure_never_starts_phase_b(self, orchestrator_env) -> None:
        """Codex #3 end-to-end: a pre-warm STORE failure aborts before
        the measured RETRIEVE pass ever issues a request."""
        pass_labels: list[str] = []

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            pass_labels.append(pass_label)
            return RequestResult(
                store_status="store_failed",
                total_chunks=2,
                failure="STORE store_failed (seq %d, cold pass)" % seq_no,
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(
                _stub_command(),
                _orchestrator_args(op="retrieve-only", requests=3),
            )
        assert excinfo.value.code == 1
        assert "warm" not in pass_labels  # Phase B never started

    def test_teardown_unregister_failure_invalidates_run(
        self, orchestrator_env
    ) -> None:
        """Codex #9: a clean load phase followed by a failed UNREGISTER
        must not be reported as a valid run (leaked server context)."""

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            return RequestResult(
                store_status="stored", store_tokens=512, total_chunks=2
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        orchestrator_env.setattr(
            sv_cmd,
            "_teardown_worker_contexts",
            lambda contexts, use_handle, log: ["UNREGISTER_KV_CACHE[0] timed out"],
        )
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(_stub_command(), _orchestrator_args())
        assert excinfo.value.code == 1

    def test_checksum_mismatch_invalidates_run(self, orchestrator_env) -> None:
        """A pair run whose oracle FAILs must exit non-zero (restores the
        historical exit-1-on-mismatch contract; also the run-level half
        of the B3 negative control)."""
        chk = {"cold": ["a" * 32], "warm": ["b" * 32]}

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            return RequestResult(
                checksums=chk[pass_label],
                store_status="stored" if pass_label == "cold" else None,
                retrieve_status="retrieved" if pass_label == "warm" else None,
                hit_chunks=2 if pass_label == "warm" else 0,
                total_chunks=2,
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(
                _stub_command(),
                _orchestrator_args(op="pair", requests=1, checksum="on"),
            )
        assert excinfo.value.code == 1


# ------------------------------------------------------------------ #
#  Two-phase start barrier (M2)                                       #
# ------------------------------------------------------------------ #


class TestStartBarrier:
    def test_no_request_before_start_gate_releases(self) -> None:
        """Codex #6: all workers must be created and ready before the
        first request is issued; the measurement hook observes zero
        completed work at release time."""
        records: list[int] = []
        seen_at_ready: list[int] = []

        def _drive(wc, stats, **kwargs):
            records.append(wc.worker_id)

        def _on_all_ready() -> None:
            seen_at_ready.append(len(records))

        num_workers = 4
        contexts = [
            _WorkerContext(
                worker_id=w, instance_id=w, client=_DUMMY_CLIENT, band_blocks=0
            )
            for w in range(num_workers)
        ]
        stats = [WorkerStats(worker_id=w) for w in range(num_workers)]
        _run_phase(
            contexts,
            _drive,
            stats_list=stats,
            num_workers=num_workers,
            start=0,
            requests_per_worker=1,
            end=None,
            interval=0.0,
            stop_event=threading.Event(),
            progress=None,
            inline=False,
            on_all_ready=_on_all_ready,
        )
        assert seen_at_ready == [0]  # gate released with zero work done
        assert sorted(records) == list(range(num_workers))

    def test_inline_path_invokes_hook_before_drive(self) -> None:
        order: list[str] = []

        def _drive(wc, stats, **kwargs):
            order.append("drive")

        _run_phase(
            [
                _WorkerContext(
                    worker_id=0, instance_id=0, client=_DUMMY_CLIENT, band_blocks=0
                )
            ],
            _drive,
            stats_list=[WorkerStats(worker_id=0)],
            num_workers=1,
            start=0,
            requests_per_worker=1,
            end=None,
            interval=0.0,
            stop_event=threading.Event(),
            progress=None,
            inline=True,
            on_all_ready=lambda: order.append("ready"),
        )
        assert order == ["ready", "drive"]


# ------------------------------------------------------------------ #
#  Per-worker deterministic seed (M1)                                 #
# ------------------------------------------------------------------ #


class TestPerWorkerSeed:
    @staticmethod
    def _alloc(seed: int) -> torch.Tensor:
        return _allocate_kv_cache(
            num_layers=1,
            num_heads=2,
            head_size=4,
            num_blocks=4,
            block_size=2,
            dtype=torch.float16,
            device="cpu",
            kv_size=2,
            seed=seed,
        )[0]

    def test_same_seed_is_reproducible(self) -> None:
        assert torch.equal(self._alloc(42), self._alloc(42))

    def test_different_worker_seeds_differ(self) -> None:
        """Codex #7: distinct workers must hold distinct KV bytes, or a
        cross-context misread could never be detected."""
        base = sv_helpers._BASE_ALLOC_SEED
        assert not torch.equal(self._alloc(base + 0), self._alloc(base + 1))

    def test_global_rng_state_untouched(self) -> None:
        torch.random.manual_seed(1234)
        state_before = torch.random.get_rng_state()
        self._alloc(99)
        assert torch.equal(state_before, torch.random.get_rng_state())


# ------------------------------------------------------------------ #
#  Handle-mode checksum oracle (B3)                                   #
# ------------------------------------------------------------------ #


class _OracleChecksumHandler(BaseHTTPRequestHandler):
    """Content-sensitive /cache/checksums stand-in.

    Hashes the *actual shared tensors* over the requested block range —
    exactly what the real endpoint does to the registered pages — so the
    test exercises the oracle's discriminative power: if the tensors
    were poisoned and nothing restored them, the digest must change.
    """

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode())
        blocks = payload["block_ids"]
        srv = self.server
        checksums = _compute_client_checksums(
            srv.kv_tensors,
            min(blocks),
            len(blocks),
            srv.block_size,
            srv.token_chunk_size,
        )
        body = json.dumps({"status": "success", "chunk_checksums": checksums}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class TestHandleModeChecksumOracle:
    """B3: in handle mode the warm pass must zero-fill the shared pages
    before RETRIEVE, so a silent no-op RETRIEVE cannot PASS."""

    BLOCK_SIZE = 16
    CHUNK_SIZE = 256  # tokens -> 16 blocks per chunk

    @pytest.fixture
    def oracle_env(self, monkeypatch):
        tensors = [
            torch.randn((2, 64, self.BLOCK_SIZE, 2, 4), dtype=torch.float16)
            for _ in range(2)
        ]
        server = HTTPServer(("127.0.0.1", 0), _OracleChecksumHandler)
        server.kv_tensors = tensors
        server.block_size = self.BLOCK_SIZE
        server.token_chunk_size = self.CHUNK_SIZE
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield tensors, "http://127.0.0.1:%d" % server.server_address[1], monkeypatch
        server.shutdown()

    def _run_pair(self, tensors, http_base, monkeypatch, retrieve_writes_back):
        snapshot = [t.clone() for t in tensors]

        def _retrieve(payloads):
            if retrieve_writes_back:
                for t, s in zip(tensors, snapshot, strict=True):
                    t.copy_(s)
            return (0, True)  # server reports success either way

        behavior = {
            RequestType.LOOKUP: None,
            RequestType.STORE: (0, True),
            RequestType.RETRIEVE: _retrieve,
        }

        def _fake_call(client, request_type, payloads, timeout_s=10.0):
            if request_type == RequestType.QUERY_PREFETCH_STATUS:
                # Cold pass misses everything, warm pass hits everything.
                request_id = payloads[0]
                return 0 if request_id.endswith("cold") else 2
            action = behavior.get(request_type)
            if callable(action):
                return action(payloads)
            return action

        monkeypatch.setattr(sv_helpers, "_call", _fake_call)
        kwargs = _request_kwargs(
            http_base=http_base,
            total_blocks=64,
            checksum=ChecksumMode.ON,
            handle_tensors=tensors,
        )
        cold = _process_request(_DUMMY_CLIENT, 0, pass_label="cold", **kwargs)
        warm = _process_request(_DUMMY_CLIENT, 0, pass_label="warm", **kwargs)
        return cold, warm

    def test_noop_retrieve_fails_checksum(self, oracle_env) -> None:
        """Negative control (codex #8): RETRIEVE reports success but
        never writes the pages back -> the digests must differ."""
        tensors, http_base, monkeypatch = oracle_env
        cold, warm = self._run_pair(
            tensors, http_base, monkeypatch, retrieve_writes_back=False
        )
        assert cold is not None and warm is not None
        assert cold.checksums and warm.checksums
        assert cold.checksums != warm.checksums
        # The poison itself is observable: the retrieved range is zero.
        assert torch.count_nonzero(tensors[0].narrow(1, 0, 32)) == 0

    def test_writeback_retrieve_passes_checksum(self, oracle_env) -> None:
        """Positive control: a RETRIEVE that restores the bytes must
        reproduce the cold digest (the oracle discriminates, not just
        rejects)."""
        tensors, http_base, monkeypatch = oracle_env
        cold, warm = self._run_pair(
            tensors, http_base, monkeypatch, retrieve_writes_back=True
        )
        assert cold is not None and warm is not None
        assert cold.checksums == warm.checksums


# ------------------------------------------------------------------ #
#  Teardown failure reporting (codex #9, unit level)                  #
# ------------------------------------------------------------------ #


class TestTeardownFailures:
    def test_unregister_timeout_reported_and_cleanup_continues(
        self, monkeypatch
    ) -> None:
        closed: list[int] = []

        class _Client:
            def __init__(self, wid: int) -> None:
                self._wid = wid

            def close(self) -> None:
                closed.append(self._wid)

        # The batch deregister acks each (client, instance_id) in order;
        # instance 0 "times out" (False), instance 1 succeeds (True).
        monkeypatch.setattr(
            sv_cmd,
            "_send_unregister_kv_cache_batch",
            lambda clients_and_ids, use_handle: [
                iid != 0 for _client, iid in clients_and_ids
            ],
        )
        contexts = [
            _WorkerContext(
                worker_id=w,
                instance_id=w,
                client=cast(MessageQueueClient, _Client(w)),
                band_blocks=0,
                registered=True,
            )
            for w in range(2)
        ]
        failures = _teardown_worker_contexts(contexts, True, lambda m: None)
        assert len(failures) == 1
        assert "UNREGISTER_KV_CACHE[0]" in failures[0]
        # One worker's failure never skips the others' cleanup.
        assert closed == [0, 1]

    def test_clean_teardown_returns_empty(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sv_cmd,
            "_send_unregister_kv_cache_batch",
            lambda clients_and_ids, use_handle: [True for _ in clients_and_ids],
        )

        class _Client:
            def close(self) -> None:
                pass

        contexts = [
            _WorkerContext(
                worker_id=0,
                instance_id=0,
                client=cast(MessageQueueClient, _Client()),
                band_blocks=0,
                registered=True,
            )
        ]
        assert _teardown_worker_contexts(contexts, True, lambda m: None) == []


# ------------------------------------------------------------------ #
#  CPU handle-mode event handle (minor fix)                           #
# ------------------------------------------------------------------ #


class TestEventHandleCpu:
    def test_cpu_event_handle_is_empty(self) -> None:
        """CPU SHM needs no cross-process CUDA event; handle mode with
        use_gpu=False must send the empty handle instead of building a
        CUDA event (regression for the lmcache_driven-on-cpu path)."""
        assert sv_helpers._make_event_handle(use_gpu=False) == b""

    def test_cpu_handle_store_sends_empty_event_handle(self, monkeypatch) -> None:
        recorded: list = []

        def _fake_call(client, request_type, payloads, timeout_s=10.0):
            recorded.append((request_type, payloads))
            return (0, True)

        monkeypatch.setattr(sv_helpers, "_call", _fake_call)
        key = _make_key((0, 9906) + (9906,) * 30, request_id="req-cpu", end=32)
        status = sv_helpers._send_store(
            _DUMMY_CLIENT, key, block_size=16, use_gpu=False, use_handle=True
        )
        assert status == "stored"
        # The event-handle payload slot carries the empty CPU handle.
        assert recorded[0][1][3] == b""


# ------------------------------------------------------------------ #
#  Per-run nonce (fresh instance_id base + request_id prefix)         #
# ------------------------------------------------------------------ #


class TestRequestIdNonce:
    """The run nonce prefixes every request_id so successive runs against
    one server never collide on its session table."""

    def test_request_id_includes_nonce(self, monkeypatch) -> None:
        seen: list[str] = []

        def _capture_lookup(payloads):
            seen.append(payloads[0].request_id)
            return None  # void = LOOKUP accepted

        behavior = {
            RequestType.LOOKUP: _capture_lookup,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss
            RequestType.STORE: (0, True),
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, []))
        _process_request(
            _DUMMY_CLIENT,
            7,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(nonce=999),
        )
        assert seen == ["req-999-7-cold"]

    def test_instance_id_base_threaded_and_in_metadata(self, orchestrator_env) -> None:
        """instance_id = base + worker_id, the same base is the request
        nonce, and the base is recorded in the config metadata."""
        recorded: list = []

        def _fake_setup(wc, **kwargs):
            recorded.append((wc.instance_id, kwargs.get("nonce")))
            wc.request_kwargs = {}
            wc.registered = False

        orchestrator_env.setattr(sv_cmd, "_setup_worker_context", _fake_setup)

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            return RequestResult(
                store_status="stored", store_tokens=512, total_chunks=2
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        cmd = _CapturingMetricsCommand()
        run_server_bench(cast(BaseCommand, cmd), _orchestrator_args())
        assert cmd.metrics is not None
        cfg = cmd.metrics.to_dict()["metrics"]["config"]
        base = cfg["instance_id_base"]
        assert isinstance(base, int) and base >= 1
        # Single worker: instance_id == base + 0, and the request nonce is
        # the same base.
        assert recorded == [(base, base)]


# ------------------------------------------------------------------ #
#  Session cleanup: END_SESSION + FREE_LOOKUP_LOCKS                   #
# ------------------------------------------------------------------ #


class TestSessionCleanup:
    """Once a LOOKUP is accepted the server holds session state (and, on a
    hit, read locks); END_SESSION must run on every exit and
    FREE_LOOKUP_LOCKS must run on a contract violation that held locks."""

    def test_end_session_on_normal_completion(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss
            RequestType.STORE: (0, True),
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert RequestType.END_SESSION in calls

    def test_end_session_on_prefetch_poll_failure(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None and "prefetch status poll failed" in result.failure
        # A poll failure still ends the session the accepted LOOKUP opened.
        assert RequestType.END_SESSION in calls
        # ...and taints the server: the prefetch job leaks (END_SESSION does
        # not cancel it), so the dedicated server must be restarted.
        assert result.server_tainted is True

    def test_end_session_timeout_taints_and_fails(self, monkeypatch) -> None:
        # An END_SESSION timeout on an otherwise-successful request both
        # taints the server and becomes a request failure, so the driver
        # aborts the worker immediately (fail-close) instead of continuing
        # to issue requests against a tainted server.
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss -> STORE
            RequestType.STORE: (0, True),
            RequestType.END_SESSION: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None
        assert result.server_tainted is True
        assert "END_SESSION timeout" in result.failure

    def test_body_exception_plus_end_session_timeout_raises_tainted(
        self, monkeypatch
    ) -> None:
        # If the body raises before a result exists and END_SESSION then
        # times out, the taint is carried out as ServerTaintedError with the
        # original body exception chained (never silently dropped).
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss -> STORE path
            RequestType.END_SESSION: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, []))

        def _boom(*args, **kwargs):
            raise RuntimeError("gather blew up")

        monkeypatch.setattr(sv_helpers, "_send_store", _boom)
        with pytest.raises(sv_helpers.ServerTaintedError) as excinfo:
            _process_request(
                _DUMMY_CLIENT,
                0,
                pass_label="cold",
                contract=RequestContract.STORE_MISS,
                **_request_kwargs(),
            )
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "gather blew up" in str(excinfo.value.__cause__)

    def test_send_end_session_reports_ack(self, monkeypatch) -> None:
        monkeypatch.setattr(
            sv_helpers, "_call", _dispatching_call({RequestType.END_SESSION: None}, [])
        )
        assert sv_helpers._send_end_session(_DUMMY_CLIENT, "req") is True
        monkeypatch.setattr(
            sv_helpers,
            "_call",
            _dispatching_call({RequestType.END_SESSION: sv_helpers._TIMEOUT}, []),
        )
        assert sv_helpers._send_end_session(_DUMMY_CLIENT, "req") is False

    def test_lookup_timeout_best_effort_ends_session(self, monkeypatch) -> None:
        calls: list = []
        behavior = {RequestType.LOOKUP: sv_helpers._TIMEOUT}
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT, 0, pass_label="cold", **_request_kwargs()
        )
        # The LOOKUP was *submitted*, so the server may have created session
        # state: END_SESSION runs best-effort (a no-op server-side if the
        # request was never accepted), and the run is tainted. No transfer or
        # lock work happens on top of the timeout.
        assert RequestType.END_SESSION in calls
        assert RequestType.FREE_LOOKUP_LOCKS not in calls
        assert result is not None and result.server_tainted is True

    def test_body_exception_with_acked_end_session_still_taints(
        self, monkeypatch
    ) -> None:
        # Producer path: an exception raised mid-request (here inside the
        # prefetch poll) leaves indeterminate server state -- the prefetch job
        # is removed only by a *completed* QUERY_PREFETCH_STATUS. A successful
        # END_SESSION clears the session but NOT that job, so _process_request
        # must still raise ServerTaintedError instead of letting the bare
        # exception escape untainted (which would report reuse safe: yes).
        calls: list = []

        def _boom(_payloads):
            raise RuntimeError("connection reset mid-poll")

        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: _boom,
            RequestType.END_SESSION: None,  # cleanup ACKS
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        with pytest.raises(sv_helpers.ServerTaintedError) as ei:
            _process_request(
                _DUMMY_CLIENT,
                0,
                pass_label="cold",
                contract=RequestContract.STORE_MISS,
                **_request_kwargs(),
            )
        # Cleanup was attempted despite the failure ...
        assert RequestType.END_SESSION in calls
        # ... the taint reason names the un-cleared side effect ...
        assert "prefetch job" in str(ei.value)
        # ... and the original exception is chained, not swallowed.
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_body_keyboardinterrupt_with_acked_end_session_taints(
        self, monkeypatch
    ) -> None:
        # Producer path, interrupt variant: a Ctrl-C during the poll must be
        # re-raised as ServerTaintedInterrupt (a KeyboardInterrupt subclass ->
        # the run still exits 130) even though END_SESSION ACKS, so the
        # orchestrator reports the server unsafe to reuse. This exercises the
        # real _process_request producer, not a stubbed final wrapper.
        calls: list = []

        def _interrupt(_payloads):
            raise KeyboardInterrupt

        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: _interrupt,
            RequestType.END_SESSION: None,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        with pytest.raises(sv_helpers.ServerTaintedInterrupt) as ei:
            _process_request(
                _DUMMY_CLIENT,
                0,
                pass_label="cold",
                contract=RequestContract.STORE_MISS,
                **_request_kwargs(),
            )
        # Interrupt semantics preserved (still a KeyboardInterrupt subclass).
        assert isinstance(ei.value, KeyboardInterrupt)
        assert RequestType.END_SESSION in calls
        assert isinstance(ei.value.__cause__, KeyboardInterrupt)

    def test_end_session_exception_taints_after_success(self, monkeypatch) -> None:
        # Producer path: END_SESSION is itself submit-then-unknown. A fully
        # successful request whose END_SESSION raises (not a timeout, which is
        # caught) leaves the session possibly still held, so _process_request
        # must raise ServerTaintedError rather than let the cleanup exception
        # escape (which would drop the taint verdict).
        calls: list = []

        def _boom(_payloads):
            raise RuntimeError("zmq send failed on END_SESSION")

        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss
            RequestType.STORE: (0, True),  # store succeeds -> result produced
            RequestType.END_SESSION: _boom,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        with pytest.raises(sv_helpers.ServerTaintedError) as ei:
            _process_request(
                _DUMMY_CLIENT,
                0,
                pass_label="cold",
                contract=RequestContract.STORE_MISS,
                **_request_kwargs(),
            )
        # The request body itself succeeded (STORE was issued and acked) ...
        assert RequestType.STORE in calls
        # ... but the cleanup failure is surfaced as a taint, chaining it.
        assert "unknown server state" in str(ei.value)
        assert isinstance(ei.value.__cause__, RuntimeError)

    def test_end_session_interrupt_taints_as_interrupt(self, monkeypatch) -> None:
        # Producer path, interrupt variant: a Ctrl-C while END_SESSION is
        # in flight must raise ServerTaintedInterrupt (a KeyboardInterrupt
        # subclass -> the run still exits 130) with reuse safe: no, not a bare
        # KeyboardInterrupt that would drop the taint.
        calls: list = []

        def _interrupt(_payloads):
            raise KeyboardInterrupt

        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,
            RequestType.STORE: (0, True),
            RequestType.END_SESSION: _interrupt,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        with pytest.raises(sv_helpers.ServerTaintedInterrupt) as ei:
            _process_request(
                _DUMMY_CLIENT,
                0,
                pass_label="cold",
                contract=RequestContract.STORE_MISS,
                **_request_kwargs(),
            )
        assert isinstance(ei.value, KeyboardInterrupt)
        assert isinstance(ei.value.__cause__, KeyboardInterrupt)

    def test_register_failure_marks_registered_for_teardown(self, monkeypatch) -> None:
        # A REGISTER that fails / times out may have created the context
        # server-side, so the worker is marked registered before the raise
        # and teardown best-effort UNREGISTERs it (submit-then-unknown).
        monkeypatch.setattr(
            sv_cmd,
            "_allocate_cpu_shm_kv_cache",
            lambda **kw: ([object()], [object()], ["shm_x"]),
        )
        monkeypatch.setattr(sv_cmd, "_send_register_kv_cache", lambda *a, **k: False)
        wc = _WorkerContext(
            worker_id=0,
            instance_id=42,
            client=_DUMMY_CLIENT,
            band_blocks=64,
        )
        with pytest.raises(RuntimeError, match="server effect unknown"):
            sv_cmd._setup_worker_context(
                wc,
                layer_groups=[],
                band_blocks=64,
                layout_hints={},
                engine_group_infos=[],
                num_engine_group_infos=1,
                use_gpu=False,
                use_handle=True,
                block_size=16,
                num_tokens=511,
                chunk_size=256,
                http_base="",
                poll_interval=0.0,
                checksum=ChecksumMode.ON,
                nonce=42,
                log=lambda m: None,
            )
        assert wc.registered is True

    def test_register_interrupt_marks_registered_for_teardown(
        self, monkeypatch
    ) -> None:
        # Producer path: submit_request() sends REGISTER before _call blocks on
        # the response, so a Ctrl-C / exception raised *while waiting* (not a
        # timeout, which is caught) means the server may hold the context with
        # no ack. wc.registered is set BEFORE the submit, so even this raise
        # leaves teardown to best-effort UNREGISTER it.
        monkeypatch.setattr(
            sv_cmd,
            "_allocate_cpu_shm_kv_cache",
            lambda **kw: ([object()], [object()], ["shm_x"]),
        )

        def _interrupt(*a, **k):
            raise KeyboardInterrupt

        monkeypatch.setattr(sv_cmd, "_send_register_kv_cache", _interrupt)
        wc = _WorkerContext(
            worker_id=0,
            instance_id=42,
            client=_DUMMY_CLIENT,
            band_blocks=64,
        )
        with pytest.raises(KeyboardInterrupt):
            sv_cmd._setup_worker_context(
                wc,
                layer_groups=[],
                band_blocks=64,
                layout_hints={},
                engine_group_infos=[],
                num_engine_group_infos=1,
                use_gpu=False,
                use_handle=True,
                block_size=16,
                num_tokens=511,
                chunk_size=256,
                http_base="",
                poll_interval=0.0,
                checksum=ChecksumMode.ON,
                nonce=42,
                log=lambda m: None,
            )
        assert wc.registered is True

    def test_free_locks_on_store_miss_violation(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 2,  # unexpected full hit
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None and "contract violation" in result.failure
        # Held read locks on the hit chunks are released, then the session
        # ends; no transfer is issued.
        assert RequestType.FREE_LOOKUP_LOCKS in calls
        assert RequestType.END_SESSION in calls
        assert RequestType.STORE not in calls
        assert RequestType.RETRIEVE not in calls

    def test_free_locks_timeout_taints_server(self, monkeypatch) -> None:
        # A FREE_LOOKUP_LOCKS timeout may leave read locks held, so the
        # contract-violation failure is joined by a server taint.
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 2,  # unexpected full hit
            RequestType.FREE_LOOKUP_LOCKS: sv_helpers._TIMEOUT,
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, []))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="cold",
            contract=RequestContract.STORE_MISS,
            **_request_kwargs(),
        )
        assert result is not None
        assert "contract violation" in result.failure
        assert result.server_tainted is True

    def test_free_locks_on_retrieve_hit_partial_violation(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 1,  # partial hit of 2
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="warm",
            contract=RequestContract.RETRIEVE_HIT,
            **_request_kwargs(),
        )
        assert result is not None and "contract violation" in result.failure
        assert RequestType.FREE_LOOKUP_LOCKS in calls
        assert RequestType.END_SESSION in calls

    def test_no_free_locks_on_retrieve_hit_full_miss(self, monkeypatch) -> None:
        calls: list = []
        behavior = {
            RequestType.LOOKUP: None,
            RequestType.QUERY_PREFETCH_STATUS: 0,  # full miss, no locks held
        }
        monkeypatch.setattr(sv_helpers, "_call", _dispatching_call(behavior, calls))
        result = _process_request(
            _DUMMY_CLIENT,
            0,
            pass_label="warm",
            contract=RequestContract.RETRIEVE_HIT,
            **_request_kwargs(),
        )
        assert result is not None and "contract violation" in result.failure
        # No locks were acquired on a full miss, so none are freed.
        assert RequestType.FREE_LOOKUP_LOCKS not in calls
        assert RequestType.END_SESSION in calls


# ------------------------------------------------------------------ #
#  Checksum fail-close (pair mode)                                    #
# ------------------------------------------------------------------ #


class TestChecksumFailClose:
    """With checksum ON and a warm hit, missing / short / mismatched
    digests must invalidate the run (fail-close), not pass silently."""

    @staticmethod
    def _wc(checksum: ChecksumMode = ChecksumMode.ON) -> _WorkerContext:
        return _WorkerContext(
            worker_id=0,
            instance_id=0,
            client=_DUMMY_CLIENT,
            band_blocks=1024,
            request_kwargs={"checksum": checksum},
        )

    def _drive(
        self,
        monkeypatch,
        cold: RequestResult,
        warm: RequestResult,
        checksum: ChecksumMode = ChecksumMode.ON,
    ) -> WorkerStats:
        results = iter([cold, warm])

        def _fake(client, seq_no, pass_label, progress=None, **kw):
            return next(results)

        monkeypatch.setattr(sv_cmd, "_process_request", _fake)
        stats = WorkerStats(worker_id=0)
        _drive_pair(
            self._wc(checksum),
            stats,
            num_workers=1,
            start=0,
            requests_per_worker=1,
            end=None,
            interval=0.0,
            stop_event=None,
            progress=None,
        )
        return stats

    @staticmethod
    def _cold(checksums) -> RequestResult:
        return RequestResult(
            checksums=checksums,
            store_status="stored",
            hit_chunks=0,
            total_chunks=2,
        )

    @staticmethod
    def _warm(checksums) -> RequestResult:
        return RequestResult(
            checksums=checksums,
            retrieve_status="retrieved",
            hit_chunks=2,
            total_chunks=2,
        )

    def test_matching_digests_pass(self, monkeypatch) -> None:
        stats = self._drive(monkeypatch, self._cold(["a", "b"]), self._warm(["a", "b"]))
        assert stats.checksum_ok == 1
        assert stats.checksum_fail == 0

    def test_mismatch_is_failure(self, monkeypatch) -> None:
        with pytest.raises(_RequestFailureError, match="checksum verification failed"):
            self._drive(monkeypatch, self._cold(["a", "b"]), self._warm(["a", "c"]))

    def test_missing_warm_digest_is_failure(self, monkeypatch) -> None:
        with pytest.raises(_RequestFailureError, match="checksum verification failed"):
            self._drive(monkeypatch, self._cold(["a", "b"]), self._warm(None))

    def test_short_digest_is_failure(self, monkeypatch) -> None:
        # Fewer chunks than expected (2) must not silently pass.
        with pytest.raises(_RequestFailureError, match="checksum verification failed"):
            self._drive(monkeypatch, self._cold(["a"]), self._warm(["a"]))

    def test_zero_hit_warm_pass_is_failure(self, monkeypatch) -> None:
        # checksum ON but the warm pass hit nothing (no retrieve, no digests):
        # the cold STORE was not retrievable, so nothing was verified. This
        # must fail-close, not tally as a clean, verified request.
        warm_miss = RequestResult(
            checksums=None,
            retrieve_status=None,
            hit_chunks=0,
            total_chunks=2,
        )
        with pytest.raises(_RequestFailureError, match="hit 0 of 2 chunks"):
            self._drive(monkeypatch, self._cold(["a", "b"]), warm_miss)

    def test_checksum_off_skips_verification(self, monkeypatch) -> None:
        # Even a mismatch is ignored when verification is OFF.
        stats = self._drive(
            monkeypatch,
            self._cold(["a", "b"]),
            self._warm(["a", "c"]),
            checksum=ChecksumMode.OFF,
        )
        assert stats.checksum_ok == 0
        assert stats.checksum_fail == 0

    # (A no-hit warm pass with checksum ON is a fail-close, covered by
    #  test_zero_hit_warm_pass_is_failure above — it is NOT a silent skip.)


# ------------------------------------------------------------------ #
#  Interrupt validity (Ctrl-C -> invalid, exit 130)                  #
# ------------------------------------------------------------------ #


class TestInterruptValidity:
    def test_ctrl_c_marks_invalid_and_exits_130(self, orchestrator_env) -> None:
        """A Ctrl-C during the load phase invalidates the run: no
        throughput / latency sections, Valid: no, exit 130."""

        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            raise KeyboardInterrupt

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        cmd = _CapturingMetricsCommand()
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(cast(BaseCommand, cmd), _orchestrator_args())
        assert excinfo.value.code == 130
        assert cmd.metrics is not None
        d = cmd.metrics.to_dict()["metrics"]
        assert d["results"]["valid"] == "no"
        assert d["results"]["error"] == "interrupted"
        # A partial / interrupted run never reports throughput or latency.
        assert "throughput" not in d
        assert "cold_store" not in d

    def test_ctrl_c_with_cleanup_timeout_reports_unsafe(self, orchestrator_env) -> None:
        # A Ctrl-C whose END_SESSION could not be acknowledged surfaces as
        # ServerTaintedInterrupt: still exit 130, but Server reuse safe: no.
        def _fake_process(client, seq_no, pass_label, progress=None, **kw):
            raise sv_helpers.ServerTaintedInterrupt(
                "interrupted and END_SESSION timed out"
            )

        orchestrator_env.setattr(sv_cmd, "_process_request", _fake_process)
        cmd = _CapturingMetricsCommand()
        with pytest.raises(SystemExit) as excinfo:
            run_server_bench(cast(BaseCommand, cmd), _orchestrator_args())
        assert excinfo.value.code == 130
        assert cmd.metrics is not None
        d = cmd.metrics.to_dict()["metrics"]
        assert d["results"]["valid"] == "no"
        assert d["results"]["server_reuse_safe"] == "no"

    def test_run_phase_threaded_reraises_keyboardinterrupt(self) -> None:
        """The threaded phase must not swallow a Ctrl-C: it stops + joins
        the workers, then re-raises so the orchestrator marks the run
        invalid. Driven here via an on_all_ready that raises on the
        orchestrator thread (the same thread a real SIGINT lands on)."""

        def _drive(wc, stats, **kwargs):
            return None

        def _on_all_ready() -> None:
            raise KeyboardInterrupt

        contexts = [
            _WorkerContext(
                worker_id=w, instance_id=w, client=_DUMMY_CLIENT, band_blocks=0
            )
            for w in range(2)
        ]
        stats = [WorkerStats(worker_id=w) for w in range(2)]
        stop = threading.Event()
        with pytest.raises(KeyboardInterrupt):
            _run_phase(
                contexts,
                _drive,
                stats_list=stats,
                num_workers=2,
                start=0,
                requests_per_worker=1,
                end=None,
                interval=0.0,
                stop_event=stop,
                progress=None,
                inline=False,
                on_all_ready=_on_all_ready,
            )
        assert stop.is_set()
