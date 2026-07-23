# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench server`` subcommand implementation.

This module provides argument registration via :func:`add_server_arguments`
and the execution orchestrator :func:`run_server_bench` for the end-to-end
LMCache MP cache-server sanity test.

The command exercises the full store / retrieve data path:

    For each request:
      1. LOOKUP   — submit prefix lookup (void reply)
      2. QUERY_PREFETCH_STATUS — poll by request_id until done
      3. RETRIEVE — for the hit portion (if any)
      4. STORE    — for the miss portion
      5. CHECKSUM — verify KV cache integrity via HTTP API

Usage examples::

    # GPU mode: real CUDA tensors + IPC
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --num-tokens 512 --start 0 --end 3

    # Custom KV cache shape (multi-group spec)
    lmcache bench server --rpc-url tcp://localhost:5555 \\
        --kvcache-shape-spec '(2,32,1024,8,128):float16:32'

    # Run forever starting from sequence 0
    lmcache bench server --rpc-url tcp://localhost:5555
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import argparse
import math
import mmap
import os
import secrets
import sys
import threading
import time

# First Party
from lmcache import torch_dev

# Heavy imports reused by the orchestrator. ``DTYPE_MAP`` is required
# for the ``--kvcache-shape-spec`` help string at parser-registration
# time. On a slim install these symbols are placeholders; the
# ``_require_full_install`` guard inside the helpers module keeps
# orchestration safe.
from lmcache.cli.commands.bench.server_bench.helpers import (
    _BASE_ALLOC_SEED,
    _DEFAULT_PREFETCH_POLL_INTERVAL_S,
    _DEFAULT_SHAPE_SPEC,
    DTYPE_MAP,
    ChecksumMode,
    OpMode,
    RequestContract,
    ServerTaintedError,
    ServerTaintedInterrupt,
    _allocate_cpu_shm_kv_cache,
    _allocate_gpu_kv_cache,
    _band_blocks,
    _get_chunk_size,
    _process_request,
    _require_full_install,
    _send_register_kv_cache,
    _send_unregister_kv_cache_batch,
    shm_open_pool_as_mmap,
)

if TYPE_CHECKING:
    # Standard
    from collections.abc import Callable, Iterator

    # First Party
    from lmcache.cli.commands.base import BaseCommand
    from lmcache.cli.commands.bench.server_bench.helpers import RequestResult
    from lmcache.cli.profiling import FlameProfiler
    from lmcache.v1.multiprocess.custom_types import KVCache
    from lmcache.v1.multiprocess.mq import MessageQueueClient


# Stash the original (full-install) ImportError so the parser-stub
# branch and the orchestrator branch can both surface it verbatim.
__all__ = (
    "add_server_arguments",
    "run_server_bench",
)

# How long the run waits for every worker thread to reach the start
# barrier before declaring the phase broken (generous: thread creation
# is cheap, this only guards against a wedged interpreter).
_START_BARRIER_TIMEOUT_S = 120.0


class _RequestFailureError(RuntimeError):
    """A request-level failure that must invalidate the whole run.

    Raised by the per-worker drivers when a request reports a non-empty
    :attr:`RequestResult.failure` (LOOKUP timeout, prefetch-poll
    failure, STORE / RETRIEVE failure or timeout, or a workload-contract
    violation). :func:`_run_phase` catches it like any worker error: the
    worker's stats record the message, the shared stop event halts the
    remaining workers, and the orchestrator marks the run invalid and
    exits non-zero.
    """


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench server`` arguments to *parser*.

    Requires the full LMCache install (torch, zmq, etc.).
    Callers should check ``_IMPORT_ERROR`` before calling this.

    Args:
        parser: The ``ArgumentParser`` for the server bench subcommand.
    """

    parser.add_argument(
        "--rpc-url",
        default="tcp://localhost:5555",
        help=("ZMQ endpoint of the MP server (default: tcp://localhost:5555)"),
    )
    parser.add_argument(
        "--mode",
        choices=["cpu", "gpu"],
        default="gpu",
        help=(
            "Run mode (default: gpu). In cpu mode the client allocates "
            "POSIX-SHM-backed KV cache tensors and the server maps the "
            "same physical pages."
        ),
    )
    parser.add_argument(
        "--transfer-mode",
        choices=["auto", "engine_driven", "lmcache_driven"],
        default="auto",
        help=(
            "Transport routing for STORE/RETRIEVE (default: auto). "
            "`lmcache_driven` forces the server-driven handle path "
            "(REGISTER_KV_CACHE + STORE/RETRIEVE), which supports "
            "both CUDA IPC and CPU SHM for zero-copy transfers. "
            "`engine_driven` forces the worker-side gather/scatter "
            "data path (REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT + "
            "PREPARE/COMMIT). "
            "`auto` keeps the historical mapping: "
            "gpu->lmcache_driven, cpu->engine_driven."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=512,
        help="Tokens per request (default: 512)",
    )

    # -- KV cache shape --
    kv = parser.add_argument_group("KV cache shape")
    kv.add_argument(
        "--kvcache-shape-spec",
        type=str,
        default=_DEFAULT_SHAPE_SPEC,
        help=(
            "KV shape spec. Describes one or more KV layer groups "
            "separated by ';'. "
            "Grammar: "
            "'(kv_size,NB,BS,NH,HS):dtype:layers[;(...):dtype:layers...]'. "
            "Fields: kv_size=2 for classical K/V or 1 for MLA, "
            "NB=num_blocks, BS=block_size (tokens/block), "
            "NH=num_heads, HS=head_size (elements). "
            "dtype is the element dtype (supported: %s); 'uint8' "
            "is used for FP8-quantized KV. 'layers' is the number "
            "of consecutive layers sharing this group's geometry. "
            "Multi-group example (MLA + classical attention): "
            "'(1,1024,16,1,128):float16:4;"
            "(2,1024,16,8,128):float16:28'. "
            "All groups must share the same NB and BS. "
            "See lmcache.v1.kv_layer_groups.parse_kvcache_shape_spec "
            "for the authoritative parser. Default: '%s'"
            % (", ".join(DTYPE_MAP.keys()), _DEFAULT_SHAPE_SPEC)
        ),
    )
    kv.add_argument(
        "--num-blocks",
        type=int,
        default=1024,
        help="Paged blocks (default: 1024)",
    )
    kv.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per block (default: 16)",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting sequence number (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help=("Ending sequence number (exclusive). If not set, runs forever."),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help=("Seconds between requests (default: 0.5)"),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help=("HTTP base URL for checksum API (default: http://localhost:8080)"),
    )

    # -- concurrency & workload --
    workload = parser.add_argument_group("concurrency & workload")
    workload.add_argument(
        "--concurrency",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of concurrent worker threads, each with its own MP "
            "client and registered context (default: 1). Each worker "
            "drives a disjoint sequence-number range and its own "
            "band-size KV cache, so workers never touch each other's "
            "data. N=1 preserves the historical single-client request "
            "topology and default workload semantics."
        ),
    )
    workload.add_argument(
        "--op",
        choices=[m.value for m in OpMode],
        default=OpMode.PAIR.value,
        help=(
            "Workload shape (default: pair). 'pair' runs the historical "
            "cold STORE pass followed by a warm RETRIEVE pass per sequence "
            "and compares checksums. 'store-only' runs a single cold STORE "
            "pass over unique sequences (write throughput). 'retrieve-only' "
            "pre-warms the cache with a STORE pass, then measures a warm "
            "RETRIEVE pass (read throughput)."
        ),
    )
    workload.add_argument(
        "--requests",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Requests per worker (default: unset). When set, each worker "
            "issues exactly M requests, so the run processes concurrency*M "
            "distinct sequences total. Mutually exclusive with --end. When "
            "unset the run uses --start/--end (or runs forever), "
            "stride-partitioned across workers."
        ),
    )
    workload.add_argument(
        "--prefetch-poll-interval",
        type=float,
        default=_DEFAULT_PREFETCH_POLL_INTERVAL_S,
        metavar="SECS",
        help=(
            "Seconds between QUERY_PREFETCH_STATUS polls while waiting for "
            "a warm LOOKUP to become ready (default: %.2f). The historical "
            "50 ms cadence dominates the per-request latency floor at high "
            "concurrency; lower it to observe the server's true prefetch "
            "latency. Must be finite and >= 0. The inter-poll sleep budget "
            "is 50 * interval; the total LOOKUP wall time also includes the "
            "per-poll RPC latency and, in the worst case, the RPC timeout."
            % _DEFAULT_PREFETCH_POLL_INTERVAL_S
        ),
    )
    workload.add_argument(
        "--checksum",
        choices=[m.value for m in ChecksumMode],
        default=ChecksumMode.AUTO.value,
        help=(
            "Checksum verification mode (default: auto). 'auto' enables "
            "checksums in pair mode and disables them in store-only / "
            "retrieve-only mode so throughput runs are not skewed by the "
            "per-request checksum round-trip. 'on' / 'off' force the choice."
        ),
    )
    workload.add_argument(
        "--server-max-gpu-workers",
        type=int,
        default=None,
        metavar="M",
        help=(
            "The AFFINITY transfer-pool size the target server was started "
            "with. The server sizes that pool from its `max_gpu_workers` "
            "setting (add_affinity_thread_pool(max_workers=max_gpu_workers)), "
            "and it does not expose the value over RPC, so supply it "
            "explicitly; it is recorded in the results config section. "
            "Required when --concurrency > 1, and the run is refused when "
            "concurrency exceeds it (the bench's N fresh identities can then "
            "no longer be guaranteed N distinct transfer slots)."
        ),
    )
    workload.add_argument(
        "--server-commit",
        default="",
        metavar="REV",
        help=(
            "Free-form identifier of the server build under test (e.g. "
            "its git commit). Recorded verbatim in the results config "
            "section so runs stay attributable."
        ),
    )
    workload.add_argument(
        "--server-image",
        default="",
        metavar="IMAGE",
        help=(
            "Free-form identifier of the server container image / "
            "environment under test. Recorded verbatim in the results "
            "config section."
        ),
    )

    prof = parser.add_argument_group(
        "server profiling",
        "Flame-graph the MP server process while this benchmark drives "
        "load into it. The server's store path (hashing, allocation, "
        "gather, D2H) runs in its own process, not in this client, so "
        "profiling attaches to --profile-server-pid rather than to the "
        "benchmark. See 'lmcache tool flamegraph' for the standalone form.",
    )
    prof.add_argument(
        "--flamegraph",
        choices=["on", "off"],
        default="off",
        help="Record a flame graph of the server during the run (default: off).",
    )
    prof.add_argument(
        "--profile-server-pid",
        type=int,
        default=0,
        metavar="PID",
        help=(
            "Server process to profile, e.g. $(pgrep -f 'lmcache server'). "
            "Required when --flamegraph on."
        ),
    )
    prof.add_argument(
        "--flamegraph-mode",
        default="gil",
        metavar="MODE[,MODE...]",
        help=(
            "What to sample in the server (default: gil). Pass several "
            "comma-separated to profile one load run per mode. Modes: on-cpu, "
            "off-cpu, wakeup, offwake (perf/bcc), wall, gil (py-spy). perf/bcc "
            "name Python functions only when the server was launched with "
            "PYTHONPERFSUPPORT=1. See the 'lmcache tool flamegraph' docs."
        ),
    )
    prof.add_argument(
        "--flamegraph-output",
        default="",
        metavar="PATH",
        help=(
            "SVG output path. Default: "
            "/tmp/lmcache_bench_flames/server-pid<PID>.<mode>.svg."
        ),
    )
    prof.add_argument(
        "--flamegraph-scripts-dir",
        default="",
        metavar="DIR",
        help=(
            "Directory with the FlameGraph scripts (flamegraph.pl, "
            "stackcollapse-perf.pl); default ~/FlameGraph (cloned there on "
            "first use). Unused by --flamegraph-mode wall / gil, which "
            "render their own SVG."
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _build_server_profiler(
    args: argparse.Namespace,
    log: "Callable[[str], None]",
) -> "FlameProfiler | None":
    """Build a profiler attached to the server, or ``None`` if disabled.

    Validates the target pid and toolchain eagerly so a misconfigured run
    fails before any load is sent. The returned profiler is not started;
    the caller wraps the load loop with ``start`` / ``stop``.

    Args:
        args: Parsed CLI arguments for ``lmcache bench server``.
        log: Progress logger.

    Returns:
        A ready :class:`FlameProfiler` targeting ``--profile-server-pid``,
        or ``None`` when ``--flamegraph`` is off.
    """
    if getattr(args, "flamegraph", "off") != "on":
        return None

    # First Party
    from lmcache.cli.profiling import (
        PY_SPY_MODES,
        FlameProfiler,
        ProfileError,
        check_profiling_deps,
        default_output_path,
        resolve_flamegraph_dir,
    )

    pid = args.profile_server_pid
    if pid <= 0:
        print(
            "Error: --flamegraph on requires --profile-server-pid "
            "(the pid of the running 'lmcache server').",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        print(f"Error: no such process: --profile-server-pid {pid}", file=sys.stderr)
        sys.exit(2)
    except PermissionError:
        print(
            f"Error: server pid {pid} belongs to another user; "
            "profiling it needs root.",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        check_profiling_deps(args.flamegraph_mode)
        flamegraph_dir = ""
        if args.flamegraph_mode not in PY_SPY_MODES:
            flamegraph_dir = resolve_flamegraph_dir(args.flamegraph_scripts_dir, log)
        output = args.flamegraph_output or default_output_path(
            f"server-pid{pid}", args.flamegraph_mode
        )
        return FlameProfiler(
            mode=args.flamegraph_mode,
            output=output,
            flamegraph_dir=flamegraph_dir,
            pid=pid,
            title=f"{args.flamegraph_mode} (server pid {pid})",
        )
    except ProfileError as e:
        print(
            "Error: --flamegraph on was requested but profiling is "
            f"unavailable:\n  {e}",
            file=sys.stderr,
        )
        sys.exit(2)


@dataclass
class _WorkloadConfig:
    """Resolved, validated concurrency / workload settings.

    Produced by :func:`_resolve_workload_config` from parsed CLI args so
    the orchestrator receives fully-typed, already-checked values.

    Args:
        op: Workload shape.
        num_workers: Concurrent worker count (``>= 1``).
        requests_per_worker: Per-worker request cap, or ``None`` when the
            run is bounded by ``--end`` / unbounded.
        checksum: Resolved checksum mode (never :attr:`ChecksumMode.AUTO`).
        poll_interval: Seconds between prefetch-status polls.
        use_handle: Effective transport path resolved from ``--mode`` +
            ``--transfer-mode``: ``True`` for the handle path (GPU CUDA-IPC
            or CPU ``lmcache_driven`` SHM handle), ``False`` for the
            engine-driven data path (worker-side gather/scatter). The data
            path is admitted only for the historical ``--op pair
            --concurrency 1`` sanity test; every throughput op and every
            concurrent run requires the handle path.
        server_max_gpu_workers: Operator-supplied AFFINITY transfer-pool
            size of the target server (its ``max_gpu_workers`` setting;
            metadata, not queryable over RPC), or ``None`` when not
            provided (single-worker runs only).
    """

    op: OpMode
    num_workers: int
    requests_per_worker: "int | None"
    checksum: ChecksumMode
    poll_interval: float
    use_handle: bool
    server_max_gpu_workers: "int | None"


def _resolve_workload_config(args: argparse.Namespace) -> _WorkloadConfig:
    """Validate and resolve the concurrency / workload CLI arguments.

    Enforces the invariants that must hold before any connection opens:
    ``--concurrency >= 1``; ``--requests >= 1`` when set; ``--requests``
    and ``--end`` are mutually exclusive; concurrent runs must be bounded;
    ``--op retrieve-only`` must be bounded; ``--prefetch-poll-interval``
    must be finite and non-negative; ``--checksum on`` is rejected for the
    throughput modes (no checksum contract is defined there yet);
    ``--server-max-gpu-workers`` must be supplied for any concurrent run
    and must not be exceeded by ``--concurrency`` (the distinct-transfer-
    slot guarantee cannot hold beyond it). It also resolves the effective
    transport path (``use_handle``) from ``--mode`` + ``--transfer-mode``
    and refuses the engine-driven data path for anything but the
    historical ``--op pair --concurrency 1`` sanity test: a throughput op
    or a concurrent run on the data path would exercise an unsupported
    (unverified, fail-open) code path, so it must use the handle path
    instead. Also resolves :attr:`ChecksumMode.AUTO` to ``ON`` in pair
    mode and ``OFF`` in the throughput modes.

    Args:
        args: Parsed CLI arguments for ``lmcache bench server``.

    Returns:
        A validated :class:`_WorkloadConfig`.

    Raises:
        ValueError: If any argument combination is invalid.
    """
    op = OpMode(getattr(args, "op", OpMode.PAIR.value))
    num_workers = getattr(args, "concurrency", 1)
    if num_workers < 1:
        raise ValueError("--concurrency must be >= 1, got %d" % num_workers)
    requests_per_worker = getattr(args, "requests", None)
    end = getattr(args, "end", None)
    if requests_per_worker is not None:
        if requests_per_worker < 1:
            raise ValueError("--requests must be >= 1, got %d" % requests_per_worker)
        if end is not None:
            raise ValueError(
                "--requests and --end are mutually exclusive stop conditions; "
                "pass one or the other."
            )
    if requests_per_worker is None and end is None and num_workers > 1:
        raise ValueError(
            "unbounded (forever) runs are only supported with "
            "--concurrency 1; pass --requests M or --end N for a bounded "
            "concurrent run."
        )
    if op is OpMode.RETRIEVE_ONLY and requests_per_worker is None and end is None:
        raise ValueError(
            "--op retrieve-only requires a bounded run (--requests M or "
            "--end N) because it pre-warms the full sequence set before "
            "measuring the retrieve pass."
        )
    server_max_gpu_workers = getattr(args, "server_max_gpu_workers", None)
    if server_max_gpu_workers is not None and server_max_gpu_workers < 1:
        raise ValueError(
            "--server-max-gpu-workers must be >= 1, got %d" % server_max_gpu_workers
        )
    if num_workers > 1 and server_max_gpu_workers is None:
        raise ValueError(
            "--concurrency > 1 requires --server-max-gpu-workers (the "
            "AFFINITY transfer-pool size the server was started with, i.e. "
            "its max_gpu_workers; it cannot be queried over RPC). It is "
            "recorded with the results and gates the run: the bench's N "
            "identities are guaranteed N distinct transfer slots only when "
            "N <= max_gpu_workers on a dedicated server."
        )
    if server_max_gpu_workers is not None and num_workers > server_max_gpu_workers:
        raise ValueError(
            "refusing to run: --concurrency %d exceeds "
            "--server-max-gpu-workers %d, so the bench workers cannot all "
            "land on distinct server transfer slots and the run would "
            "measure slot contention instead of N-way concurrency. Reduce "
            "--concurrency or restart the server with a larger pool."
            % (num_workers, server_max_gpu_workers)
        )
    checksum_arg = ChecksumMode(getattr(args, "checksum", ChecksumMode.AUTO.value))
    # The throughput modes have no checksum contract yet: store-only never
    # reads back what it wrote, and retrieve-only's pre-warm STORE is not
    # verified. Rather than silently ignore ``--checksum on`` there, reject
    # it so the operator is not misled into thinking a run was verified.
    # A follow-up may add a store->independent-read verification pass.
    if checksum_arg is ChecksumMode.ON and op is not OpMode.PAIR:
        raise ValueError(
            "--checksum on is only supported with --op pair; the "
            "store-only / retrieve-only throughput modes have no checksum "
            "contract yet (a follow-up will add an independent read-back "
            "verification pass). Use --op pair, or drop --checksum on."
        )
    if checksum_arg is ChecksumMode.AUTO:
        checksum = ChecksumMode.ON if op is OpMode.PAIR else ChecksumMode.OFF
    else:
        checksum = checksum_arg
    poll_interval = getattr(
        args, "prefetch_poll_interval", _DEFAULT_PREFETCH_POLL_INTERVAL_S
    )
    if not (math.isfinite(poll_interval) and poll_interval >= 0):
        raise ValueError(
            "--prefetch-poll-interval must be finite and >= 0, got %r"
            % (poll_interval,)
        )
    # Resolve the effective transport path from mode + transfer_mode. This
    # mirrors the historical mapping run_server_bench used to compute
    # inline: auto -> gpu means the handle path / cpu means the data path;
    # explicit lmcache_driven forces the handle path; explicit
    # engine_driven forces the data path.
    use_gpu = getattr(args, "mode", "gpu") == "gpu"
    transfer_mode = getattr(args, "transfer_mode", "auto")
    if transfer_mode == "auto":
        use_handle = use_gpu
    elif transfer_mode == "lmcache_driven":
        use_handle = True
    else:
        use_handle = False
    # NARROW SCOPE: the engine-driven data path (use_handle=False) is only
    # admitted for the historical `--op pair --concurrency 1` sanity test.
    # Throughput ops and concurrent runs on the data path would exercise an
    # unverified, fail-open code path (the scatter/gather self-check is only
    # a discriminating oracle at N=1 pair). Engine-driven concurrency /
    # throughput is a deferred follow-up; require the handle path instead.
    if not use_handle and (op is not OpMode.PAIR or num_workers > 1):
        raise ValueError(
            "the engine-driven data path (--transfer-mode engine_driven, or "
            "--mode cpu without --transfer-mode lmcache_driven) is only "
            "supported for the historical `--op pair --concurrency 1` "
            "sanity test. Requested op=%s concurrency=%d needs the handle "
            "path: use --mode gpu, or --transfer-mode lmcache_driven on cpu. "
            "Engine-driven throughput / concurrency is a follow-up."
            % (op.value, num_workers)
        )
    if not use_handle:
        # The engine-driven data path is admitted ONLY as the historical
        # N=1 pair sanity test, and only fully guarded:
        #   * CPU only — GPU + engine_driven is not a validated path.
        #   * --checksum on — the client-side scatter/gather self-check is
        #     the *only* oracle that can catch a dropped copy or an empty
        #     COMMIT payload on this path (the server still ACKs them), so
        #     running it unverified would be silently fail-open.
        # Everything else (concurrency, throughput, native transport) lives
        # on the handle path; engine-driven concurrency is a follow-up.
        if use_gpu:
            raise ValueError(
                "--transfer-mode engine_driven is only supported on "
                "--mode cpu; on gpu use the handle path (drop "
                "--transfer-mode, or use --transfer-mode lmcache_driven)."
            )
        if checksum is not ChecksumMode.ON:
            raise ValueError(
                "the engine-driven data path (--mode cpu without "
                "--transfer-mode lmcache_driven) requires --checksum on: "
                "its client-side self-check is the only oracle that can "
                "detect a dropped scatter/gather copy or an empty commit "
                "payload, so it may not run unverified. Use --checksum on, "
                "or switch to the handle path (--transfer-mode "
                "lmcache_driven)."
            )
    return _WorkloadConfig(
        op=op,
        num_workers=num_workers,
        requests_per_worker=requests_per_worker,
        checksum=checksum,
        poll_interval=poll_interval,
        use_handle=use_handle,
        server_max_gpu_workers=server_max_gpu_workers,
    )


@dataclass
class _WorkerContext:
    """Per-worker registered resources for the concurrent bench topology.

    Under the one-client-plus-one-context-per-worker design each worker
    owns its own :class:`MessageQueueClient` (a distinct zmq DEALER
    identity -> a distinct affinity key, letting the server distribute
    the ``ThreadPoolType.AFFINITY`` STORE / RETRIEVE handlers across its
    transfer workers instead of serializing all bench workers onto one.
    The server maps a key to a slot on the key's first appearance,
    round-robin over its pool, and never frees slots — so N distinct
    slots are guaranteed only on a dedicated server, with the bench's N
    identities appearing fresh and N <= the server's max_gpu_workers; see
    ``--server-max-gpu-workers``) and its own registered KV cache context
    (``instance_id = instance_base + worker_id`` for a per-run nonce
    ``instance_base``, so a fresh run practically avoids colliding with a
    prior run's still-registered context) sized to a single band
    (``total_blocks // N``). The total registered KV tensor capacity
    therefore never exceeds the requested pool and stays approximately
    constant as concurrency grows (floor division may leave up to N-1
    blocks unused, e.g. ``1000 // 3 * 3 = 999``); per-client and
    per-context metadata overhead still scales with N.

    Args:
        worker_id: Zero-based worker index.
        instance_id: The registered context ID this worker owns.
        client: This worker's dedicated MP client.
        band_blocks: Paged blocks in this worker's own KV cache (its band).
        request_kwargs: Fixed keyword arguments forwarded to
            :func:`_process_request` for every request this worker issues.
        registered: True once REGISTER_KV_CACHE has been *submitted* (set
            before the call, since the server may create the context even on
            a timeout or an interrupt while awaiting the reply), so teardown
            best-effort UNREGISTERs it. UNREGISTER of a context the server
            never created is a safe no-op, so this may be True for a context
            that does not actually exist server-side.
        server_pool: Data-mode mmap of the server SHM pool (``None`` in
            handle mode).
        shm_names: POSIX-SHM segment names to unlink on teardown (CPU mode).
        keepalive: Tensors / wrappers held alive for the process lifetime so
            the server's IPC / SHM mappings are not reclaimed mid-run.
    """

    worker_id: int
    instance_id: int
    client: "MessageQueueClient"
    band_blocks: int
    request_kwargs: dict = field(default_factory=dict)
    registered: bool = False
    server_pool: "mmap.mmap | None" = None
    shm_names: list[str] = field(default_factory=list)
    keepalive: tuple = ()


def run_server_bench(
    command: "BaseCommand",
    args: argparse.Namespace,
) -> None:
    """Centralized orchestrator: run the server bench loop.

    Builds one MP client and one registered KV-cache context per worker
    (see :class:`_WorkerContext`), drives the selected workload, tears the
    contexts down, and emits the metrics summary. ``--op retrieve-only``
    runs a two-phase global barrier: an unmeasured STORE pre-warm across
    all workers, a join, and only then a measured RETRIEVE pass (so the
    pre-warm is excluded from the wall time and the server profiler);
    the measured pass starts only when the pre-warm provably completed
    (:func:`_prewarm_gate`). Concurrent phases start behind a barrier:
    the wall clock and the profiler start once every worker is ready,
    before the gate releases them together. A worker exception — which
    includes any request-level failure, timeout, or workload-contract
    violation (:class:`_RequestFailureError`) — fails the whole run: the
    first exception sets the shared stop event, the remaining workers
    drain, teardown runs, and the process exits non-zero with a summary
    marked invalid. Checksum mismatches and failed teardown UNREGISTERs
    also invalidate the run.

    Args:
        command: The owning :class:`BaseCommand` instance, used to
            obtain a configured :class:`Metrics` object via
            ``command.create_metrics``.
        args: Parsed CLI arguments for ``lmcache bench server``.
    """
    _require_full_install()

    # Heavy imports — safe now that _require_full_install passed.
    # Third Party
    import zmq

    # First Party
    from lmcache.v1.kv_layer_groups import (
        format_kvcache_shape_spec,
        parse_kvcache_shape_spec,
    )
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.mq import MessageQueueClient

    quiet = getattr(args, "quiet", False)

    def log(msg: str) -> None:
        """Print progress messages; suppressed by --quiet."""
        if not quiet:
            print(msg)

    use_gpu = args.mode == "gpu"
    if use_gpu and not torch_dev.is_available():
        print("ERROR: --mode gpu requires CUDA", file=sys.stderr)
        sys.exit(1)

    # Build the profiler before opening any connection so a bad pid or a
    # missing toolchain fails immediately. ``None`` when --flamegraph is off.
    profiler = _build_server_profiler(args, log)
    profiler_started = False

    # Resolve and validate the concurrency / workload configuration up
    # front so a bad combination fails before any connection is opened.
    # The effective transport path (data vs handle) is resolved here too,
    # and the engine-driven data path is refused for anything beyond the
    # historical `--op pair --concurrency 1` sanity test.
    workload = _resolve_workload_config(args)
    op = workload.op
    num_workers = workload.num_workers
    requests_per_worker = workload.requests_per_worker
    checksum = workload.checksum
    poll_interval = workload.poll_interval
    use_handle = workload.use_handle
    if use_handle and not use_gpu:
        log(
            "  [info] --transfer-mode=lmcache_driven on cpu mode: "
            "using REGISTER_KV_CACHE + STORE/RETRIEVE over POSIX SHM"
        )

    # Run nonce: a fresh positive base computed once per run. Each worker's
    # registered context id is instance_base + worker_id, and the same base
    # prefixes every request_id. The server NOOP-re-registers an already
    # present instance_id (it refreshes liveness but does not replace the
    # tensors), so reusing a fixed id across runs would silently inherit a
    # prior run's stale context; a fresh 62-bit base per run practically
    # avoids that (it is a probabilistic guard, not a hard guarantee).
    instance_base = secrets.randbits(62) + 1

    url = args.rpc_url
    log("Connecting to LMCache MP Server at %s (mode=%s) ..." % (url, args.mode))

    ctx = zmq.Context()

    # Accumulators visible after the try/finally so the summary is emitted
    # whether the run finished, aborted, or was interrupted.
    worker_contexts: list[_WorkerContext] = []
    worker_stats: list[WorkerStats] = []
    load_wall_s = 0.0
    bytes_per_token = 0
    load_start = 0.0
    run_error = ""
    interrupted = False
    interrupt_tainted = False
    teardown_failures: list[str] = []

    try:
        # One client per worker: each DEALER socket gets a distinct zmq
        # identity (a distinct affinity key), so the server can spread the
        # workers across its transfer pool instead of one shared lane
        # (distinct slots require a dedicated server with
        # max_gpu_workers >= N; enforced via --server-max-gpu-workers).
        # Each worker's context id is the run nonce + its worker index, so
        # ids are fresh per run (no stale NOOP re-registration) and still
        # distinct across workers within a run.
        for w in range(num_workers):
            client = MessageQueueClient(url, ctx)
            worker_contexts.append(
                _WorkerContext(
                    worker_id=w,
                    instance_id=instance_base + w,
                    client=client,
                    band_blocks=0,
                )
            )

        # Query chunk size from server (any client will do).
        chunk_size = _get_chunk_size(worker_contexts[0].client)
        log("Server chunk_size = %d" % chunk_size)

        # Parse KV shape spec.
        layer_groups = parse_kvcache_shape_spec(args.kvcache_shape_spec)
        num_engine_group_infos = len(layer_groups) or 1
        log("Resolved KV shape spec: %s" % format_kvcache_shape_spec(layer_groups))

        # Paged KV demands identical NB / BS across all groups.
        first = layer_groups[0]
        nb_vals = {g.shape_desc.nb for g in layer_groups}
        bs_vals = {g.shape_desc.bs for g in layer_groups}
        if len(nb_vals) > 1 or len(bs_vals) > 1:
            raise ValueError(
                "All groups must share NB and BS (paged KV "
                "requires uniform block geometry). Got NB=%s BS=%s"
                % (sorted(nb_vals), sorted(bs_vals))
            )
        num_layers = sum(g.num_layers for g in layer_groups)
        spec_nb = getattr(first.shape_desc, "nb", 0) or 0
        spec_bs = getattr(first.shape_desc, "bs", 0) or 0
        total_blocks = spec_nb if spec_nb > 0 else args.num_blocks
        block_size = spec_bs if spec_bs > 0 else args.block_size
        if spec_nb and spec_nb != args.num_blocks:
            log(
                "  [info] spec nb=%d overrides --num-blocks=%d"
                % (spec_nb, args.num_blocks)
            )
        if spec_bs and spec_bs != args.block_size:
            log(
                "  [info] spec bs=%d overrides --block-size=%d"
                % (spec_bs, args.block_size)
            )

        # Display / legacy hint fields only.
        heads_set = {g.shape_desc.nh for g in layer_groups}
        hs_set = {g.shape_desc.hs for g in layer_groups}
        kv_size_set = {g.shape_desc.kv_size for g in layer_groups}
        dtype_set = {g.dtype for g in layer_groups}
        num_heads_disp: int | str = (
            first.shape_desc.nh if len(heads_set) == 1 else "mixed"
        )
        head_size_disp: int | str = first.shape_desc.hs if len(hs_set) == 1 else "mixed"
        kv_size_disp: int | str = (
            first.shape_desc.kv_size if len(kv_size_set) == 1 else "mixed"
        )
        if len(dtype_set) == 1:
            dtype_str = next(
                (k for k, v in DTYPE_MAP.items() if v == first.dtype),
                "float16",
            )
        else:
            dtype_str = "mixed"

        # Per-worker band size. Each worker registers its own band-size KV
        # cache, so the block space is partitioned across workers without
        # inflating total memory. Fails fast if the pool cannot be divided.
        num_tokens = args.num_tokens
        num_full_tokens_req = ((num_tokens + 1) // chunk_size) * chunk_size
        num_req_blocks = num_full_tokens_req // block_size
        band_blocks = _band_blocks(total_blocks, num_workers, num_req_blocks)
        for wc in worker_contexts:
            wc.band_blocks = band_blocks

        # dtype is sent as a string; per-layer fields are "mixed" for
        # heterogeneous specs. num_blocks reflects the per-worker band.
        layout_hints = {
            "num_layers": num_layers,
            "num_heads": num_heads_disp,
            "head_size": head_size_disp,
            "num_blocks": band_blocks,
            "block_size": block_size,
            "dtype": dtype_str,
        }
        engine_group_infos = [
            EngineGroupInfo(
                engine_group_id=group_idx,
                layer_indices=tuple(group.layer_indices),
                tokens_per_block=block_size,
            )
            for group_idx, group in enumerate(layer_groups)
        ]

        log(
            "Each request: %d tokens (%d full chunks)"
            % (num_tokens + 1, (num_tokens + 1) // chunk_size)
        )
        log(
            "KV shape: %d layers, %s heads x %s, "
            "dtype=%s, blocks=%dx%d, kv=%s"
            % (
                num_layers,
                num_heads_disp,
                head_size_disp,
                dtype_str,
                total_blocks,
                block_size,
                kv_size_disp,
            )
        )
        if num_workers > 1:
            log(
                "Per-worker band: %d blocks (total %d / %d workers)"
                % (band_blocks, total_blocks, num_workers)
            )

        # Per-token wire size across all registered layers, for bytes/s.
        bytes_per_token = _bytes_per_token(layer_groups)
        http_base = args.url.rstrip("/")

        # Allocate + register one context per worker. Any failure here
        # raises; the finally block tears down whatever was set up so far.
        for wc in worker_contexts:
            _setup_worker_context(
                wc,
                layer_groups=layer_groups,
                band_blocks=band_blocks,
                layout_hints=layout_hints,
                engine_group_infos=engine_group_infos,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                block_size=block_size,
                num_tokens=num_tokens,
                chunk_size=chunk_size,
                http_base=http_base,
                poll_interval=poll_interval,
                checksum=checksum,
                nonce=instance_base,
                log=log,
            )
        log("")

        log(
            "Workload: op=%s concurrency=%d checksum=%s"
            % (op.value, num_workers, checksum.value)
        )
        log("")

        stop_event = threading.Event()
        inline = num_workers == 1
        worker_stats = [WorkerStats(worker_id=w) for w in range(num_workers)]

        def _start_measured_phase() -> None:
            """Start the profiler + wall clock once every worker is ready.

            Runs on the orchestrator thread after all workers reached the
            start barrier and before the gate is released, so thread
            startup skew is excluded from the measured wall time and the
            profiler sees only overlapped load.
            """
            nonlocal load_start, profiler_started
            if profiler is not None:
                profiler.start(log)
                profiler_started = True
            load_start = time.monotonic()

        try:
            if op is OpMode.RETRIEVE_ONLY:
                # --- Phase A: pre-warm (STORE), unmeasured, no timer/profiler.
                log(
                    "Phase A: pre-warming %d worker(s) (STORE, unmeasured)..."
                    % num_workers
                )
                prewarm_stats = [WorkerStats(worker_id=w) for w in range(num_workers)]
                _run_phase(
                    worker_contexts,
                    _drive_store_pass,
                    stats_list=prewarm_stats,
                    num_workers=num_workers,
                    start=args.start,
                    requests_per_worker=requests_per_worker,
                    end=args.end,
                    interval=args.interval,
                    stop_event=stop_event,
                    progress=None,
                    inline=inline,
                )
                # Global barrier: all workers have joined here. Phase B
                # only starts when the pre-warm provably completed:
                # every worker finished all its expected requests and
                # every one of them was a full-miss successful STORE.
                prewarm_error = _prewarm_gate(
                    prewarm_stats,
                    start=args.start,
                    num_workers=num_workers,
                    requests_per_worker=requests_per_worker,
                    end=args.end,
                )
                if prewarm_error:
                    run_error = "pre-warm failed: %s" % prewarm_error
                    worker_stats = prewarm_stats
                    log("  [error] %s" % run_error)
                else:
                    # --- Phase B: measured RETRIEVE. Timer + profiler wrap
                    # only this phase; the pre-warm is fully excluded.
                    log(
                        "Phase B: measuring RETRIEVE across %d worker(s)..."
                        % num_workers
                    )
                    time.sleep(args.interval)  # let the server finish indexing
                    phase_error = _run_phase(
                        worker_contexts,
                        _drive_retrieve_pass,
                        stats_list=worker_stats,
                        num_workers=num_workers,
                        start=args.start,
                        requests_per_worker=requests_per_worker,
                        end=args.end,
                        interval=args.interval,
                        stop_event=stop_event,
                        progress=None,
                        inline=inline,
                        on_all_ready=_start_measured_phase,
                    )
                    if phase_error and not run_error:
                        run_error = phase_error
                    if load_start > 0.0:
                        load_wall_s = time.monotonic() - load_start
            else:
                # pair / store-only: a single measured pass. Only pair mode
                # (latency-oriented) prints per-request progress; the
                # throughput modes and --quiet stay silent.
                progress = log if (op is OpMode.PAIR and not quiet) else None
                drive = _drive_pair if op is OpMode.PAIR else _drive_store_pass
                phase_error = _run_phase(
                    worker_contexts,
                    drive,
                    stats_list=worker_stats,
                    num_workers=num_workers,
                    start=args.start,
                    requests_per_worker=requests_per_worker,
                    end=args.end,
                    interval=args.interval,
                    stop_event=stop_event,
                    progress=progress,
                    inline=inline,
                    on_all_ready=_start_measured_phase,
                )
                if phase_error and not run_error:
                    run_error = phase_error
                if load_start > 0.0:
                    load_wall_s = time.monotonic() - load_start
        except KeyboardInterrupt as exc:
            print("\nStopping...", file=sys.stderr)
            stop_event.set()
            interrupted = True
            if isinstance(exc, ServerTaintedInterrupt):
                interrupt_tainted = True
            if load_start > 0.0:
                load_wall_s = time.monotonic() - load_start
    except KeyboardInterrupt as exc:
        print("\nStopping...", file=sys.stderr)
        interrupted = True
        if isinstance(exc, ServerTaintedInterrupt):
            interrupt_tainted = True
    finally:
        # Stop recording once load ends, before teardown.
        if profiler is not None and profiler_started:
            profiler.stop(log)
        # Deregister + close every worker context while the clients are
        # still connected, then terminate the shared zmq context.
        teardown_failures = _teardown_worker_contexts(worker_contexts, use_handle, log)
        ctx.term()

    # Emit the structured metrics summary and set the exit status.
    merged = _merge_worker_stats(worker_stats)
    expected_workers = num_workers
    completed_workers = sum(1 for s in worker_stats if not s.error)
    if not run_error and merged.error:
        run_error = merged.error
    if not run_error and merged.checksum_fail > 0:
        # A checksum mismatch is a correctness failure, not a metric:
        # the run must fail loudly (restores the historical
        # exit-1-on-mismatch contract).
        run_error = "%d checksum mismatch(es)" % merged.checksum_fail
    if not run_error and teardown_failures:
        # A leaked / stuck context skews every subsequent run against
        # this server; surface it instead of pretending the run was
        # clean.
        run_error = "teardown: %s" % "; ".join(teardown_failures)
    # The server is unsafe to reuse whenever this run may have left residual
    # server-side state a nonce cannot dodge: a request-level taint (leaked
    # prefetch job or unacknowledged END_SESSION / FREE_LOOKUP_LOCKS), a
    # teardown UNREGISTER that did not complete (context may still be held),
    # or a Ctrl-C whose cleanup could not be acknowledged. Any of these must
    # be reported as ``Server reuse safe: no``.
    server_tainted = (
        merged.server_tainted or bool(teardown_failures) or interrupt_tainted
    )
    server_reuse_safe = not server_tainted
    if not run_error and merged.server_tainted:
        run_error = "server tainted (residual server-side state; restart required)"
    # A Ctrl-C anywhere invalidates the run: even the interactive
    # forever-mode (N=1 unbounded) run may show partial diagnostics but
    # must report Valid: no. It takes precedence as the run_error only when
    # nothing more specific was already recorded.
    if interrupted and not run_error:
        run_error = "interrupted"
    invalid = bool(run_error) or completed_workers < expected_workers or interrupted
    if invalid:
        log("  [warning] run invalid: %s" % (run_error or "worker(s) did not complete"))
    _emit_server_bench_metrics(
        command=command,
        args=args,
        stats=merged,
        op=op,
        num_workers=num_workers,
        wall_s=load_wall_s,
        bytes_per_token=bytes_per_token,
        expected_workers=expected_workers,
        completed_workers=completed_workers,
        run_error=run_error,
        invalid=invalid,
        instance_id_base=instance_base,
        server_reuse_safe=server_reuse_safe,
    )
    # An interrupted run exits 130 (128 + SIGINT) to distinguish an
    # operator Ctrl-C from a genuine failure (exit 1).
    if interrupted:
        sys.exit(130)
    if invalid:
        sys.exit(1)
    log("Done.")


def _setup_worker_context(
    wc: _WorkerContext,
    *,
    layer_groups: list,
    band_blocks: int,
    layout_hints: dict,
    engine_group_infos: list,
    num_engine_group_infos: int,
    use_gpu: bool,
    use_handle: bool,
    block_size: int,
    num_tokens: int,
    chunk_size: int,
    http_base: str,
    poll_interval: float,
    checksum: ChecksumMode,
    nonce: int,
    log: "Callable[[str], None]",
) -> None:
    """Allocate + register one worker's band-size KV cache context.

    Mutates *wc* in place: allocates a band-size KV cache (GPU CUDA tensors
    or CPU POSIX-SHM tensors), registers it under ``wc.instance_id``, maps
    the server SHM pool (data mode), and builds ``wc.request_kwargs``. On a
    failed registration it raises so the orchestrator's teardown reclaims
    whatever was already set up (this worker's client and any created SHM
    segments are already tracked on *wc*).

    Args:
        wc: The worker context to populate (client and instance_id set).
        layer_groups: Parsed KV layer groups (shared across workers).
        band_blocks: Per-worker paged-block count.
        layout_hints: Register-time layout hints (shared).
        engine_group_infos: Per-group metadata (shared).
        num_engine_group_infos: Number of block-id lists per transfer.
        use_gpu: Whether GPU (CUDA IPC) tensors are used.
        use_handle: Whether the handle path (REGISTER_KV_CACHE) is used.
        block_size: Tokens per paged block.
        num_tokens: Tokens per synthetic request.
        chunk_size: Server chunk size.
        http_base: HTTP base URL for the checksum API (handle mode).
        poll_interval: QUERY_PREFETCH_STATUS poll cadence.
        checksum: Resolved checksum mode.
        nonce: Per-run identity prefix for every ``request_id`` this worker
            issues (the shared run nonce that also seeds ``instance_id``).
        log: Progress logger.

    Raises:
        RuntimeError: If REGISTER_KV_CACHE fails (RPC timeout / rejection).
    """
    if use_gpu:
        # First Party
        from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

        allocated = _allocate_gpu_kv_cache(
            groups=layer_groups,
            nb_override=band_blocks,
            seed=_BASE_ALLOC_SEED + wc.worker_id,
        )
        log(
            "Allocated %d GPU tensors on %s (band=%d blocks)"
            % (len(allocated), allocated[0].device, band_blocks)
        )
        kv_wrappers: "KVCache" = [CudaIPCWrapper(t) for t in allocated]
        client_kv_tensors = allocated
        wc.keepalive = (allocated, kv_wrappers)
    else:
        # First Party
        from lmcache.v1.platform.cpu.shm import CpuShmTensorWrapper

        # Per-worker prefix so concurrent workers never collide on a segment.
        shm_prefix = "%s%d_w%d" % (
            CpuShmTensorWrapper.SHM_NAME_PREFIX,
            os.getpid(),
            wc.worker_id,
        )
        cpu_tensors, cpu_wrappers, shm_names = _allocate_cpu_shm_kv_cache(
            groups=layer_groups,
            shm_prefix=shm_prefix,
            nb_override=band_blocks,
            seed=_BASE_ALLOC_SEED + wc.worker_id,
        )
        wc.shm_names = shm_names
        log(
            "Allocated %d CPU SHM tensors (prefix=%s, band=%d blocks)"
            % (len(cpu_tensors), shm_prefix, band_blocks)
        )
        kv_wrappers = list(cpu_wrappers)
        client_kv_tensors = cpu_tensors
        wc.keepalive = (cpu_tensors, cpu_wrappers)

    # REGISTER is a stateful RPC whose effect can outlive our knowledge of
    # it: once submitted, the server may create the context even if we never
    # observe an ack -- an RPC timeout, or a Ctrl-C / exception raised while
    # blocked on the response (submit_request has already sent it by then).
    # Mark the worker registered *before* submitting, so every failure path,
    # including an interrupt during the wait, leaves teardown to best-effort
    # UNREGISTER it. UNREGISTER of an instance the server never created is a
    # safe no-op on both the handle and engine-driven paths, which is why
    # register failure does not distinguish timeout from rejection here.
    wc.registered = True
    register_result = _send_register_kv_cache(
        wc.client,
        instance_id=wc.instance_id,
        layout_hints=layout_hints,
        kv_caches=kv_wrappers if use_handle else None,
        use_gpu=use_gpu,
        use_handle=use_handle,
        engine_group_infos=engine_group_infos,
    )
    if not register_result:
        log("REGISTER_KV_CACHE[%d]: FAIL" % wc.instance_id)
        raise RuntimeError(
            "REGISTER_KV_CACHE failed for instance %d (RPC timeout or "
            "server rejection); server effect unknown" % wc.instance_id
        )
    log("REGISTER_KV_CACHE[%d]: OK" % wc.instance_id)

    # In data mode the register reply carries the server SHM pool name /
    # size; map it so STORE / RETRIEVE can exchange tensor data via slot
    # descriptors instead of round-tripping pickle through the RPC layer.
    server_pool: "mmap.mmap | None" = None
    if not use_handle and not isinstance(register_result, bool):
        shm_name = getattr(register_result, "shm_name", "")
        pool_size = getattr(register_result, "pool_size", 0)
        if shm_name and pool_size > 0:
            server_pool = shm_open_pool_as_mmap(shm_name, pool_size)
    wc.server_pool = server_pool

    # In data mode the server has no paged kv_tensors view to hash, so the
    # bench self-checks on the client (``client_tensors``). Handle mode
    # keeps the server-side /cache/checksums digest but still needs the
    # client-side references (``handle_tensors``): the warm pass
    # zero-fills the shared pages before RETRIEVE so the digest is a
    # discriminating oracle — without the poison, a silent no-op RETRIEVE
    # would hash the never-disturbed pages and falsely PASS.
    client_tensors = None if use_handle else client_kv_tensors
    handle_tensors = client_kv_tensors if use_handle else None

    wc.request_kwargs = {
        "num_tokens": num_tokens,
        "chunk_size": chunk_size,
        "http_base": http_base,
        "block_size": block_size,
        "total_blocks": band_blocks,
        "num_engine_group_infos": num_engine_group_infos,
        "use_gpu": use_gpu,
        "use_handle": use_handle,
        "client_tensors": client_tensors,
        "handle_tensors": handle_tensors,
        "server_pool": server_pool,
        "worker_id": wc.worker_id,
        "instance_id": wc.instance_id,
        "nonce": nonce,
        "poll_interval": poll_interval,
        "checksum": checksum,
    }


def _teardown_worker_contexts(
    worker_contexts: list[_WorkerContext],
    use_handle: bool,
    log: "Callable[[str], None]",
) -> list[str]:
    """Deregister and release every worker context, best-effort.

    Each worker's UNREGISTER, server-pool unmap, client close, and SHM
    unlink runs under its own guard so one worker's failure never skips the
    cleanup of the others. UNREGISTER runs before the client is closed so
    the server drops the registration (and the CUDA-IPC / POSIX-SHM
    mappings it holds) instead of leaking one context entry per bench run.

    Args:
        worker_contexts: The contexts to tear down.
        use_handle: Whether the handle-mode UNREGISTER protocol is used.
        log: Progress logger.

    Returns:
        Human-readable descriptions of every UNREGISTER that failed or
        timed out (empty when teardown was clean). The caller folds them
        into the run validity: a context the server still holds skews
        every subsequent run, so it must not be silent.
    """
    # Third Party
    import zmq

    failures: list[str] = []
    # Deregister every context concurrently against one shared deadline.
    # Serially calling _send_unregister_kv_cache would block the full RPC
    # timeout per worker against a dead/unresponsive server (~timeout * N);
    # submitting all UNREGISTERs first and then awaiting the replies caps
    # the worst case at ~one timeout for the whole batch.
    registered = [wc for wc in worker_contexts if wc.registered]
    if registered:
        try:
            acks = _send_unregister_kv_cache_batch(
                [(wc.client, wc.instance_id) for wc in registered],
                use_handle=use_handle,
            )
        except zmq.ZMQError as exc:
            acks = [False] * len(registered)
            print(
                "  [warning] UNREGISTER_KV_CACHE batch failed: %s" % exc,
                file=sys.stderr,
            )
        for wc, ok in zip(registered, acks, strict=True):
            log(
                "UNREGISTER_KV_CACHE[%d]: %s" % (wc.instance_id, "OK" if ok else "FAIL")
            )
            if not ok:
                failures.append("UNREGISTER_KV_CACHE[%d] timed out" % wc.instance_id)
    for wc in worker_contexts:
        if wc.server_pool is not None:
            try:
                wc.server_pool.close()
            except (BufferError, ValueError):
                pass
        try:
            wc.client.close()
        except zmq.ZMQError:
            pass
        for name in wc.shm_names:
            try:
                # First Party
                from lmcache.v1.platform.cpu.shm import shm_unlink

                shm_unlink(name)
            except OSError:
                pass
    return failures


@dataclass
class WorkerStats:
    """Latency samples and counters collected by a single worker.

    Every worker fills its own instance (never shared), so no locking is
    needed; :func:`_merge_worker_stats` combines them after all workers
    finish. All ``*_ms`` lists hold per-operation latencies in
    milliseconds. ``store_tokens`` / ``retrieve_tokens`` accumulate only
    *successfully* transferred token counts, so the bytes/s metric never
    counts a failed or timed-out transfer. The ``*_attempted`` /
    ``*_ok`` / ``*_failed`` / ``*_timeout`` counters record the outcome of
    every attempted STORE / RETRIEVE. ``error`` is a non-empty
    ``"<type>: <msg>"`` string if the worker aborted.

    Args:
        worker_id: Zero-based worker index; ``-1`` marks a merged result.
    """

    worker_id: int
    cold_lookup_ms: list[float] = field(default_factory=list)
    cold_store_ms: list[float] = field(default_factory=list)
    warm_lookup_ms: list[float] = field(default_factory=list)
    warm_retrieve_ms: list[float] = field(default_factory=list)
    total_requests: int = 0
    checksum_ok: int = 0
    checksum_fail: int = 0
    store_tokens: int = 0
    retrieve_tokens: int = 0
    store_attempted: int = 0
    store_ok: int = 0
    store_failed: int = 0
    store_timeout: int = 0
    retrieve_attempted: int = 0
    retrieve_ok: int = 0
    retrieve_failed: int = 0
    retrieve_timeout: int = 0
    error: str = ""
    # Set when a request may have left unrecoverable server-side state
    # (a leaked prefetch job on a poll timeout, or an unacknowledged
    # END_SESSION / FREE_LOOKUP_LOCKS). A nonce avoids id collisions but
    # not resource leaks, so the server must be restarted before reuse.
    server_tainted: bool = False


def _bytes_per_token(layer_groups: list) -> int:
    """Total KV bytes for one token position across all registered layers.

    Sums ``kv_size * num_heads * head_size * dtype.itemsize`` over every
    layer of every group, so the throughput section can convert
    transferred token counts into bytes/s.

    Args:
        layer_groups: Parsed ``KVLayerGroupInfo`` list from
            ``parse_kvcache_shape_spec``. Each element exposes a
            ``shape_desc`` (``kv_size``/``nh``/``hs``/``nl``) and a
            ``dtype``.

    Returns:
        Bytes per token summed over all layers (0 if ``layer_groups`` is
        empty).
    """
    total = 0
    for g in layer_groups:
        sd = g.shape_desc
        total += sd.kv_size * sd.nh * sd.hs * g.dtype.itemsize * sd.nl
    return total


def _account_result(stats: WorkerStats, result: "RequestResult") -> None:
    """Fold one request's STORE / RETRIEVE outcomes into failure counters.

    Pure aggregation over :attr:`RequestResult.store_status` /
    :attr:`RequestResult.retrieve_status`: increments the attempted counter
    for each operation the pass actually issued, then routes it to the
    ``ok`` / ``failed`` / ``timeout`` bucket. A ``None`` status means the
    operation was not attempted this pass (full hit or full miss) and is
    not counted.

    Args:
        stats: The worker's stats, mutated in place.
        result: The completed request result.
    """
    if result.store_status is not None:
        stats.store_attempted += 1
        if result.store_status == "stored":
            stats.store_ok += 1
        elif result.store_status == "timeout":
            stats.store_timeout += 1
        else:
            stats.store_failed += 1
    if result.retrieve_status is not None:
        stats.retrieve_attempted += 1
        if result.retrieve_status == "retrieved":
            stats.retrieve_ok += 1
        elif result.retrieve_status == "timeout":
            stats.retrieve_timeout += 1
        else:
            stats.retrieve_failed += 1
    # Taint is sticky and independent of failure: an END_SESSION timeout on
    # an otherwise-successful request leaves no ``failure`` but still leaks
    # server state, so it must be recorded here (not only on the abort path).
    if result.server_tainted:
        stats.server_tainted = True


def _record_cold_pass(stats: WorkerStats, result: "RequestResult") -> None:
    """Fold a cold-pass (STORE) result into *stats*.

    Args:
        stats: The worker's stats, mutated in place.
        result: The cold-pass request result.
    """
    if result.lookup_ms is not None:
        stats.cold_lookup_ms.append(result.lookup_ms)
    if result.store_ms is not None:
        stats.cold_store_ms.append(result.store_ms)
    stats.store_tokens += result.store_tokens
    _account_result(stats, result)


def _record_warm_pass(stats: WorkerStats, result: "RequestResult") -> None:
    """Fold a warm-pass (RETRIEVE) result into *stats*.

    Args:
        stats: The worker's stats, mutated in place.
        result: The warm-pass request result.
    """
    if result.lookup_ms is not None:
        stats.warm_lookup_ms.append(result.lookup_ms)
    if result.retrieve_ms is not None:
        stats.warm_retrieve_ms.append(result.retrieve_ms)
    stats.retrieve_tokens += result.retrieve_tokens
    _account_result(stats, result)


def _worker_seq_numbers(
    start: int,
    worker_id: int,
    num_workers: int,
    requests_per_worker: "int | None",
    end: "int | None",
    stop_event: "threading.Event | None",
) -> "Iterator[int]":
    """Yield the sequence numbers a single worker must process.

    Workers interleave with stride ``num_workers``: worker ``w`` yields
    ``start + w``, ``start + w + num_workers``, ... so every worker owns a
    disjoint set of sequence numbers (hence disjoint cache keys). With
    ``num_workers == 1`` and ``worker_id == 0`` this yields
    ``start, start + 1, ...`` — identical to the historical loop.

    Stop conditions, in priority order:

    * ``stop_event`` set (cooperative cancellation) -> stop;
    * ``requests_per_worker`` reached -> stop after that many yields;
    * ``end`` reached -> stop when ``seq_no >= end``;
    * otherwise run forever.

    Args:
        start: Base sequence number (``--start``).
        worker_id: Zero-based worker index.
        num_workers: Total worker count (stride).
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.
        stop_event: Cooperative cancellation flag, or ``None`` for the
            inline single-worker path (which relies on ``KeyboardInterrupt``).

    Yields:
        The next sequence number for this worker.
    """
    k = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            return
        if requests_per_worker is not None and k >= requests_per_worker:
            return
        seq_no = start + worker_id + k * num_workers
        if end is not None and seq_no >= end:
            return
        yield seq_no
        k += 1


def _expected_worker_requests(
    start: int,
    worker_id: int,
    num_workers: int,
    requests_per_worker: "int | None",
    end: "int | None",
) -> int:
    """Number of requests a worker must process in a bounded run.

    Mirrors the stop conditions of :func:`_worker_seq_numbers` (without
    the cooperative stop event): ``requests_per_worker`` when set,
    otherwise the size of the stride-``num_workers`` slice of
    ``[start, end)`` owned by *worker_id*.

    Args:
        start: Base sequence number (``--start``).
        worker_id: Zero-based worker index.
        num_workers: Total worker count (stride).
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.

    Returns:
        The expected request count for this worker.

    Raises:
        ValueError: If neither ``requests_per_worker`` nor ``end`` bound
            the run (an unbounded run has no expected count).
    """
    if requests_per_worker is not None:
        return requests_per_worker
    if end is None:
        raise ValueError("unbounded run has no expected request count")
    first = start + worker_id
    if first >= end:
        return 0
    return (end - first + num_workers - 1) // num_workers


def _prewarm_gate(
    prewarm_stats: list[WorkerStats],
    *,
    start: int,
    num_workers: int,
    requests_per_worker: "int | None",
    end: "int | None",
) -> str:
    """Decide whether the retrieve-only measured phase may start.

    The pre-warm is only sufficient when every worker (a) reported no
    error, (b) completed all the requests its sequence slice expects,
    and (c) stored every one of them successfully (the pre-warm drives
    :attr:`RequestContract.STORE_MISS`, so a hit, failure, or timeout
    already aborts the worker — the counts here are the belt-and-braces
    check that nothing completed short).

    Args:
        prewarm_stats: Per-worker stats filled by the pre-warm phase.
        start: Base sequence number (``--start``).
        num_workers: Total worker count.
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.

    Returns:
        ``""`` when Phase B may start, otherwise a human-readable
        reason; the caller marks the run invalid and never starts the
        measured phase.
    """
    for s in prewarm_stats:
        if s.error:
            return "worker %d: %s" % (s.worker_id, s.error)
        if s.server_tainted:
            # A pre-warm that tainted the server (leaked prefetch job or an
            # unacknowledged cleanup RPC) must not start the measured phase:
            # Phase B uses fresh stats, so the taint would otherwise be lost
            # and the summary would wrongly report Server reuse safe: yes.
            return (
                "worker %d tainted the server during pre-warm "
                "(residual server-side state; restart required)" % s.worker_id
            )
        expected = _expected_worker_requests(
            start, s.worker_id, num_workers, requests_per_worker, end
        )
        if s.total_requests != expected:
            return "worker %d completed %d/%d pre-warm requests" % (
                s.worker_id,
                s.total_requests,
                expected,
            )
        if s.store_ok != expected:
            return "worker %d stored %d/%d pre-warm requests successfully" % (
                s.worker_id,
                s.store_ok,
                expected,
            )
    return ""


def _drive_pair(
    wc: _WorkerContext,
    stats: WorkerStats,
    *,
    num_workers: int,
    start: int,
    requests_per_worker: "int | None",
    end: "int | None",
    interval: float,
    stop_event: "threading.Event | None",
    progress: "Callable[[str], None] | None",
) -> None:
    """Drive the historical PAIR workload for one worker.

    Per sequence: a cold STORE pass, an ``interval`` sleep, a warm
    RETRIEVE pass, a checksum comparison, and a trailing ``interval``
    sleep. Mutates *stats* in place so a ``KeyboardInterrupt`` on the
    inline path still leaves partial results visible. Any request-level
    failure (LOOKUP timeout, prefetch-poll failure, STORE / RETRIEVE
    failure or timeout) is recorded and then raised as
    :class:`_RequestFailureError`, invalidating the run; only the
    historical sub-chunk skip (``None`` result) continues.

    Args:
        wc: This worker's context (client + request kwargs).
        stats: The worker's stats, mutated in place.
        num_workers: Total worker count (sequence-space stride).
        start: Base sequence number.
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.
        interval: Seconds slept between the two passes and after each pair.
        stop_event: Cooperative cancellation flag (threaded path) or
            ``None`` (inline path).
        progress: Per-request progress sink, or ``None`` to stay silent.
    """
    seq_numbers = _worker_seq_numbers(
        start, wc.worker_id, num_workers, requests_per_worker, end, stop_event
    )
    for seq_no in seq_numbers:
        if progress is not None:
            progress("=== [w%d] Request seq=%d ===" % (wc.worker_id, seq_no))
        cold_result = _process_request(
            wc.client,
            seq_no,
            pass_label="cold",
            progress=progress,
            contract=RequestContract.PAIR,
            **wc.request_kwargs,
        )
        if cold_result is None:
            raise _RequestFailureError(
                "[w%d seq %d] request skipped (fewer tokens than one chunk); "
                "a pair pass that stores/verifies nothing is invalid"
                % (wc.worker_id, seq_no)
            )
        _record_cold_pass(stats, cold_result)
        if cold_result.failure:
            raise _RequestFailureError(
                "[w%d seq %d cold] %s" % (wc.worker_id, seq_no, cold_result.failure)
            )

        time.sleep(interval)

        warm_result = _process_request(
            wc.client,
            seq_no,
            pass_label="warm",
            progress=progress,
            contract=RequestContract.PAIR,
            **wc.request_kwargs,
        )
        if warm_result is None:
            raise _RequestFailureError(
                "[w%d seq %d] warm request skipped; a pair pass that "
                "retrieves/verifies nothing is invalid" % (wc.worker_id, seq_no)
            )
        _record_warm_pass(stats, warm_result)
        if warm_result.failure:
            raise _RequestFailureError(
                "[w%d seq %d warm] %s" % (wc.worker_id, seq_no, warm_result.failure)
            )

        stats.total_requests += 1

        # Checksum fail-close: when verification is ON and the warm pass
        # actually retrieved data (a hit), the cold and warm digests must
        # both be present, cover the full expected chunk count, be equal in
        # length, and match element-for-element. Anything short of that — a
        # missing / None / short digest, a length mismatch, or any unequal
        # element — is a correctness failure that invalidates the run, not a
        # silently skipped comparison. (The historical code only compared
        # when both lists were truthy and merely counted mismatches, so a
        # dropped digest passed as a clean run.)
        checksum_mode = wc.request_kwargs.get("checksum", ChecksumMode.ON)
        if checksum_mode is ChecksumMode.ON:
            expected = warm_result.total_chunks
            # In pair mode the cold pass just stored this sequence, so the
            # warm pass must hit *every* chunk. A short / zero hit means the
            # just-stored data was not retrievable (eviction, a silent store
            # no-op, or a real bug): with checksum ON that is a verification
            # failure, not a request that is silently left unverified. Guard
            # this before the digest compare so a zero-hit warm pass (no warm
            # checksums at all) can no longer pass as a clean, "verified" run.
            if warm_result.hit_chunks < expected:
                stats.checksum_fail += 1
                print(
                    "  [w%d seq %d] CHECKSUM VERIFICATION FAILED "
                    "(warm hit %d/%d chunks; nothing was verified)"
                    % (
                        wc.worker_id,
                        seq_no,
                        warm_result.hit_chunks,
                        expected,
                    ),
                    file=sys.stderr,
                )
                raise _RequestFailureError(
                    "[w%d seq %d] checksum verification failed (fail-close): "
                    "the warm pass hit %d of %d chunks, so the cold STORE was "
                    "not fully retrievable and nothing could be verified"
                    % (wc.worker_id, seq_no, warm_result.hit_chunks, expected)
                )
            cold_checksums = cold_result.checksums
            warm_checksums = warm_result.checksums
            verified = (
                cold_checksums is not None
                and warm_checksums is not None
                and len(cold_checksums) == expected
                and len(warm_checksums) == expected
                and cold_checksums == warm_checksums
            )
            if verified:
                stats.checksum_ok += 1
                if progress is not None:
                    progress(
                        "  [w%d seq %d] CHECKSUM MATCH OK" % (wc.worker_id, seq_no)
                    )
            else:
                stats.checksum_fail += 1
                # A failed / missing digest is a correctness signal, not
                # chatter: always surface it (and any per-chunk diff) on
                # stderr, then abort the worker so the run is invalid.
                print(
                    "  [w%d seq %d] CHECKSUM VERIFICATION FAILED "
                    "(cold=%s warm=%s, expected %d chunks)"
                    % (
                        wc.worker_id,
                        seq_no,
                        "none" if cold_checksums is None else len(cold_checksums),
                        "none" if warm_checksums is None else len(warm_checksums),
                        expected,
                    ),
                    file=sys.stderr,
                )
                if cold_checksums is not None and warm_checksums is not None:
                    for i, (c, w) in enumerate(
                        zip(cold_checksums, warm_checksums, strict=False)
                    ):
                        print(
                            "    chunk %d: cold=%s warm=%s %s"
                            % (i, c[:12], w[:12], ("OK" if c == w else "FAIL")),
                            file=sys.stderr,
                        )
                raise _RequestFailureError(
                    "[w%d seq %d] checksum verification failed (fail-close): "
                    "the warm RETRIEVE did not reproduce the cold STORE "
                    "digests" % (wc.worker_id, seq_no)
                )

        time.sleep(interval)


def _drive_store_pass(
    wc: _WorkerContext,
    stats: WorkerStats,
    *,
    num_workers: int,
    start: int,
    requests_per_worker: "int | None",
    end: "int | None",
    interval: float,
    stop_event: "threading.Event | None",
    progress: "Callable[[str], None] | None",
) -> None:
    """Drive a single cold STORE pass for one worker.

    Used both by ``--op store-only`` and by the ``--op retrieve-only``
    pre-warm. No inter-request sleep (write-throughput path). ``interval``
    is accepted for a uniform driver signature and ignored. Declares
    :attr:`RequestContract.STORE_MISS`: every request must be a full
    miss and STORE successfully, and never issues a RETRIEVE — any
    violation, failure, or timeout raises :class:`_RequestFailureError`
    and invalidates the run.

    Args:
        wc: This worker's context.
        stats: The worker's stats, mutated in place.
        num_workers: Total worker count (sequence-space stride).
        start: Base sequence number.
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.
        interval: Ignored (present for signature uniformity).
        stop_event: Cooperative cancellation flag, or ``None``.
        progress: Per-request progress sink, or ``None`` to stay silent.
    """
    del interval
    seq_numbers = _worker_seq_numbers(
        start, wc.worker_id, num_workers, requests_per_worker, end, stop_event
    )
    for seq_no in seq_numbers:
        if progress is not None:
            progress("=== [w%d] Request seq=%d (store) ===" % (wc.worker_id, seq_no))
        result = _process_request(
            wc.client,
            seq_no,
            pass_label="cold",
            progress=progress,
            contract=RequestContract.STORE_MISS,
            **wc.request_kwargs,
        )
        if result is None:
            raise _RequestFailureError(
                "[w%d seq %d] request skipped (fewer tokens than one chunk); "
                "a store throughput pass that transfers nothing is invalid"
                % (wc.worker_id, seq_no)
            )
        _record_cold_pass(stats, result)
        stats.total_requests += 1
        if result.failure:
            raise _RequestFailureError(
                "[w%d seq %d store] %s" % (wc.worker_id, seq_no, result.failure)
            )


def _drive_retrieve_pass(
    wc: _WorkerContext,
    stats: WorkerStats,
    *,
    num_workers: int,
    start: int,
    requests_per_worker: "int | None",
    end: "int | None",
    interval: float,
    stop_event: "threading.Event | None",
    progress: "Callable[[str], None] | None",
) -> None:
    """Drive a single warm RETRIEVE pass for one worker.

    The measured phase of ``--op retrieve-only``; assumes the sequences
    were already stored by the pre-warm pass. No inter-request sleep.
    ``interval`` is accepted for a uniform driver signature and ignored.
    Declares :attr:`RequestContract.RETRIEVE_HIT`: every request must be
    a full hit and RETRIEVE successfully, and never issues a STORE — any
    violation, failure, or timeout raises :class:`_RequestFailureError`
    and invalidates the run.

    Args:
        wc: This worker's context.
        stats: The worker's stats, mutated in place.
        num_workers: Total worker count (sequence-space stride).
        start: Base sequence number.
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.
        interval: Ignored (present for signature uniformity).
        stop_event: Cooperative cancellation flag, or ``None``.
        progress: Per-request progress sink, or ``None`` to stay silent.
    """
    del interval
    seq_numbers = _worker_seq_numbers(
        start, wc.worker_id, num_workers, requests_per_worker, end, stop_event
    )
    for seq_no in seq_numbers:
        if progress is not None:
            progress("=== [w%d] Request seq=%d (retrieve) ===" % (wc.worker_id, seq_no))
        result = _process_request(
            wc.client,
            seq_no,
            pass_label="warm",
            progress=progress,
            contract=RequestContract.RETRIEVE_HIT,
            **wc.request_kwargs,
        )
        if result is None:
            raise _RequestFailureError(
                "[w%d seq %d] request skipped (fewer tokens than one chunk); "
                "a retrieve throughput pass that transfers nothing is invalid"
                % (wc.worker_id, seq_no)
            )
        _record_warm_pass(stats, result)
        stats.total_requests += 1
        if result.failure:
            raise _RequestFailureError(
                "[w%d seq %d retrieve] %s" % (wc.worker_id, seq_no, result.failure)
            )


def _run_phase(
    worker_contexts: list[_WorkerContext],
    drive_fn: "Callable[..., None]",
    *,
    stats_list: list[WorkerStats],
    num_workers: int,
    start: int,
    requests_per_worker: "int | None",
    end: "int | None",
    interval: float,
    stop_event: threading.Event,
    progress: "Callable[[str], None] | None",
    inline: bool,
    on_all_ready: "Callable[[], None] | None" = None,
) -> str:
    """Run one workload phase across all workers, filling *stats_list*.

    ``inline`` runs the single worker on the calling thread so a
    ``KeyboardInterrupt`` propagates directly (preserving the historical
    forever-run Ctrl-C behaviour). Otherwise one thread per worker runs
    concurrently. A worker exception is captured into that worker's
    ``error`` field and sets *stop_event*, so the first failure cooperatively
    halts the remaining workers rather than letting a broken run continue.

    Threaded runs use a two-phase start: every worker thread is created
    and blocks on a start gate; once all workers have signalled ready (a
    barrier), *on_all_ready* runs on the calling thread (the orchestrator
    starts its wall clock / profiler there), and only then is the gate
    released — so the measured phase has true N-way overlap from the
    first request and thread-startup skew is excluded. The inline path
    calls *on_all_ready* immediately before driving the single worker.

    Args:
        worker_contexts: Per-worker contexts (index == worker id).
        drive_fn: The per-worker driver (``_drive_pair`` /
            ``_drive_store_pass`` / ``_drive_retrieve_pass``).
        stats_list: Pre-created per-worker stats, filled in place so
            partial results survive an interrupt.
        num_workers: Number of workers.
        start: Base sequence number.
        requests_per_worker: Per-worker request cap, or ``None``.
        end: Exclusive global sequence bound, or ``None``.
        interval: Inter-pass sleep (driver-dependent).
        stop_event: Shared cooperative cancellation flag.
        progress: Per-request progress sink, or ``None``.
        inline: Whether to run the single worker inline (no threads).
        on_all_ready: Callback run once every worker is ready and before
            any request is issued, or ``None``.

    Returns:
        A phase-level error string when starting the measured phase
        itself failed (the *on_all_ready* callback raised), or ``""``
        when the phase started cleanly. Per-worker failures are reported
        through ``stats_list[*].error`` instead, not through this value.

    Raises:
        KeyboardInterrupt: On Ctrl-C. Both the inline and the threaded
            paths propagate it (the threaded path first stops and joins the
            workers) so the orchestrator marks the run invalid and exits
            130 rather than reporting a partial run as complete.
    """
    if inline:
        if on_all_ready is not None:
            try:
                on_all_ready()
            except Exception as exc:  # noqa: BLE001 — invalid run, not a crash
                phase_error = "phase start failed: %s: %s" % (
                    type(exc).__name__,
                    exc,
                )
                print("  [error] %s" % phase_error, file=sys.stderr)
                return phase_error
        # Run on the calling thread so a KeyboardInterrupt propagates
        # directly (forever-run Ctrl-C). A genuine worker error is still
        # captured into the stats — matching the threaded path — so N=1
        # failures produce the same invalid summary + non-zero exit.
        try:
            drive_fn(
                worker_contexts[0],
                stats_list[0],
                num_workers=num_workers,
                start=start,
                requests_per_worker=requests_per_worker,
                end=end,
                interval=interval,
                stop_event=None,
                progress=progress,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 — surface as an invalid run
            stats_list[0].error = "%s: %s" % (type(exc).__name__, exc)
            # A ServerTaintedError means cleanup could not confirm the
            # server is safe to reuse, so carry the taint into the stats.
            if isinstance(exc, ServerTaintedError):
                stats_list[0].server_tainted = True
            print(
                "  [worker 0] ERROR: %s" % stats_list[0].error,
                file=sys.stderr,
            )
        return ""

    ready_barrier = threading.Barrier(num_workers + 1)
    start_event = threading.Event()

    def _target(worker_id: int) -> None:
        try:
            # Two-phase start: signal ready, then block on the gate so
            # every worker issues its first request only after the
            # orchestrator has started the phase clock.
            ready_barrier.wait(timeout=_START_BARRIER_TIMEOUT_S)
            start_event.wait()
            drive_fn(
                worker_contexts[worker_id],
                stats_list[worker_id],
                num_workers=num_workers,
                start=start,
                requests_per_worker=requests_per_worker,
                end=end,
                interval=interval,
                stop_event=stop_event,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001 — surface, don't crash siblings
            stats_list[worker_id].error = "%s: %s" % (type(exc).__name__, exc)
            # A ServerTaintedError means cleanup could not confirm the
            # server is safe to reuse, so carry the taint into the stats.
            if isinstance(exc, ServerTaintedError):
                stats_list[worker_id].server_tainted = True
            # First failure fails the run: stop the healthy workers early.
            stop_event.set()
            print(
                "  [worker %d] ERROR: %s" % (worker_id, stats_list[worker_id].error),
                file=sys.stderr,
            )

    threads = [
        threading.Thread(target=_target, args=(w,), name="bench-worker-%d" % w)
        for w in range(num_workers)
    ]
    for t in threads:
        t.start()
    phase_error = ""
    try:
        try:
            ready_barrier.wait(timeout=_START_BARRIER_TIMEOUT_S)
        except threading.BrokenBarrierError:
            # A worker died before it was ready; abort the phase. The
            # workers released below observe stop_event before their
            # first request.
            stop_event.set()
        else:
            if on_all_ready is not None:
                try:
                    on_all_ready()
                except Exception as exc:  # noqa: BLE001
                    # Starting the measured phase (e.g. profiler.start)
                    # failed. Catch it here so control still reaches the
                    # start_event.set() below: the worker threads are
                    # non-daemon and park on start_event.wait() with no
                    # timeout, so letting this escape would hang the
                    # interpreter at shutdown. Abort so the released
                    # workers issue nothing, and mark the run invalid
                    # instead of letting a broken phase report throughput.
                    phase_error = "phase start failed: %s: %s" % (
                        type(exc).__name__,
                        exc,
                    )
                    stop_event.set()
                    print("  [error] %s" % phase_error, file=sys.stderr)
        # Reached on every non-KeyboardInterrupt path (on_all_ready can no
        # longer escape), so the gate is always released and the workers
        # never park forever. On an abort, stop_event was already set
        # above, so the released workers return before their first request.
        start_event.set()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nStopping workers...", file=sys.stderr)
        stop_event.set()
        ready_barrier.abort()
        start_event.set()
        for t in threads:
            t.join()
        # Re-raise once the workers are cleanly joined so the orchestrator
        # sees the interrupt and marks the run invalid (exit 130). Swallowing
        # it here would let a partial run report throughput as if it had
        # completed.
        raise
    return phase_error


def _merge_worker_stats(stats_list: list[WorkerStats]) -> WorkerStats:
    """Combine per-worker stats into a single aggregate.

    Latency lists are concatenated (percentiles are computed over the
    pooled samples) and all counters summed. The first non-empty ``error``
    encountered is carried into the merged result.

    Args:
        stats_list: Per-worker :class:`WorkerStats`.

    Returns:
        A merged :class:`WorkerStats` with ``worker_id == -1``.
    """
    merged = WorkerStats(worker_id=-1)
    for s in stats_list:
        merged.cold_lookup_ms.extend(s.cold_lookup_ms)
        merged.cold_store_ms.extend(s.cold_store_ms)
        merged.warm_lookup_ms.extend(s.warm_lookup_ms)
        merged.warm_retrieve_ms.extend(s.warm_retrieve_ms)
        merged.total_requests += s.total_requests
        merged.checksum_ok += s.checksum_ok
        merged.checksum_fail += s.checksum_fail
        merged.store_tokens += s.store_tokens
        merged.retrieve_tokens += s.retrieve_tokens
        merged.store_attempted += s.store_attempted
        merged.store_ok += s.store_ok
        merged.store_failed += s.store_failed
        merged.store_timeout += s.store_timeout
        merged.retrieve_attempted += s.retrieve_attempted
        merged.retrieve_ok += s.retrieve_ok
        merged.retrieve_failed += s.retrieve_failed
        merged.retrieve_timeout += s.retrieve_timeout
        merged.server_tainted = merged.server_tainted or s.server_tainted
        if s.error and not merged.error:
            merged.error = s.error
    return merged


def _emit_server_bench_metrics(
    command: "BaseCommand",
    args: argparse.Namespace,
    *,
    stats: WorkerStats,
    op: OpMode,
    num_workers: int,
    wall_s: float,
    bytes_per_token: int,
    expected_workers: int,
    completed_workers: int,
    run_error: str,
    invalid: bool,
    instance_id_base: int,
    server_reuse_safe: bool = True,
) -> None:
    """Emit the server bench summary using the CLI metrics system.

    Beyond the config / results / per-operation latency sections it adds
    failure accounting (attempted / ok / failed / timeouts for STORE and
    RETRIEVE), a completed-vs-expected worker count, and a throughput
    section whose MB/s divides *successful* bytes by the load-phase wall
    time. When *invalid* is set (a worker aborted, or fewer workers
    completed than expected) it emits only the config + a results section
    marked invalid and omits the throughput / latency sections, so a
    broken run is never reported as a normal success.

    Args:
        command: The owning :class:`BaseCommand` instance.
        args: Parsed CLI arguments.
        stats: Merged :class:`WorkerStats` across all workers.
        op: The workload shape that produced *stats*.
        num_workers: Concurrency level used for the run.
        wall_s: Wall-clock seconds of the measured load phase.
        bytes_per_token: KV bytes per token, from :func:`_bytes_per_token`.
        expected_workers: Workers the run was configured to launch.
        completed_workers: Workers that finished without an error.
        run_error: First worker / phase error, or ``""``.
        invalid: Whether the run is invalid (skips success-format output).
        instance_id_base: The run nonce used as the base of every worker's
            ``instance_id`` (``base + worker_id``). Recorded so a result
            file is attributable to the exact registered contexts it used.
    """
    if stats.total_requests == 0 and not invalid:
        return

    metrics = command.create_metrics("Server Bench Result", args, width=64)

    cfg_section = metrics.add_section("config", "Configuration")
    cfg_section.add("rpc_url", "RPC URL", args.rpc_url)
    cfg_section.add("mode", "Mode", args.mode)
    cfg_section.add(
        "transfer_mode", "Transfer mode", getattr(args, "transfer_mode", "auto")
    )
    cfg_section.add("op", "Op", op.value)
    cfg_section.add("concurrency", "Concurrency", num_workers)
    cfg_section.add("num_tokens", "Tokens / request", args.num_tokens)
    cfg_section.add("interval", "Interval (s)", args.interval)
    # The run nonce that seeded every worker's registered context id, so a
    # result stays attributable to the exact contexts it registered.
    cfg_section.add("instance_id_base", "Instance ID base", instance_id_base)
    # Operator-supplied server metadata, so a result file stays
    # attributable to a specific server configuration. The AFFINITY pool
    # size can only be "unknown" on single-worker runs (required for N > 1).
    server_max_gpu_workers = getattr(args, "server_max_gpu_workers", None)
    cfg_section.add(
        "server_max_gpu_workers",
        "Server max GPU workers",
        server_max_gpu_workers if server_max_gpu_workers is not None else "unknown",
    )
    server_commit = getattr(args, "server_commit", "")
    if server_commit:
        cfg_section.add("server_commit", "Server commit", server_commit)
    server_image = getattr(args, "server_image", "")
    if server_image:
        cfg_section.add("server_image", "Server image", server_image)

    result_section = metrics.add_section("results", "Results")
    result_section.add("valid", "Valid", "no" if invalid else "yes")
    # Whether the dedicated server is safe to reuse for the next run, or
    # must be restarted because this run may have left residual state
    # (a leaked prefetch job, or unacknowledged session / lock cleanup).
    result_section.add(
        "server_reuse_safe", "Server reuse safe", "yes" if server_reuse_safe else "no"
    )
    result_section.add("completed_workers", "Completed workers", completed_workers)
    result_section.add("expected_workers", "Expected workers", expected_workers)
    result_section.add("total_requests", "Total requests", stats.total_requests)
    if run_error:
        result_section.add("error", "Error", run_error)

    # Failure accounting: only the operations the workload actually issued.
    if stats.store_attempted > 0:
        result_section.add("store_attempted", "Store attempted", stats.store_attempted)
        result_section.add("store_ok", "Store OK", stats.store_ok)
        result_section.add("store_failed", "Store failed", stats.store_failed)
        result_section.add("store_timeout", "Store timeouts", stats.store_timeout)
    if stats.retrieve_attempted > 0:
        result_section.add(
            "retrieve_attempted", "Retrieve attempted", stats.retrieve_attempted
        )
        result_section.add("retrieve_ok", "Retrieve OK", stats.retrieve_ok)
        result_section.add("retrieve_failed", "Retrieve failed", stats.retrieve_failed)
        result_section.add(
            "retrieve_timeout", "Retrieve timeouts", stats.retrieve_timeout
        )

    # Checksum rows only when comparisons actually ran (pair mode).
    checksums_compared = stats.checksum_ok + stats.checksum_fail
    if op is OpMode.PAIR and checksums_compared > 0:
        result_section.add("checksum_ok", "Checksum OK", stats.checksum_ok)
        result_section.add("checksum_fail", "Checksum FAIL", stats.checksum_fail)
        pass_rate = stats.checksum_ok / stats.total_requests * 100
        result_section.add("pass_rate", "Pass rate (%)", round(pass_rate, 2))

    if invalid:
        # Do not report a normal success-format throughput / latency for a
        # run that did not complete cleanly.
        metrics.emit()
        return

    # Throughput. ops/s and MB/s use the measured-phase wall time and only
    # successful bytes (failed / timed-out transfers do not count).
    thr_section = metrics.add_section("throughput", "Throughput")
    thr_section.add("wall_s", "Wall time (s)", round(wall_s, 3))
    store_bytes = stats.store_tokens * bytes_per_token
    retrieve_bytes = stats.retrieve_tokens * bytes_per_token
    thr_section.add(
        "successful_store_mb", "Successful store (MB)", round(store_bytes / 1e6, 3)
    )
    thr_section.add(
        "successful_retrieve_mb",
        "Successful retrieve (MB)",
        round(retrieve_bytes / 1e6, 3),
    )
    if wall_s > 0:
        thr_section.add(
            "ops_per_s", "Requests/s", round(stats.total_requests / wall_s, 2)
        )
        thr_section.add(
            "store_mb_s", "Store MB/s", round(store_bytes / 1e6 / wall_s, 2)
        )
        thr_section.add(
            "retrieve_mb_s", "Retrieve MB/s", round(retrieve_bytes / 1e6 / wall_s, 2)
        )

    # Per-operation latency summary (cold + warm passes).
    _add_latency_section(
        metrics, "cold_lookup", "Cold Lookup (ms)", stats.cold_lookup_ms
    )
    _add_latency_section(metrics, "cold_store", "Cold Store (ms)", stats.cold_store_ms)
    _add_latency_section(
        metrics, "warm_lookup", "Warm Lookup (ms)", stats.warm_lookup_ms
    )
    _add_latency_section(
        metrics, "warm_retrieve", "Warm Retrieve (ms)", stats.warm_retrieve_ms
    )

    metrics.emit()


def _add_latency_section(
    metrics,
    section_id: str,
    section_title: str,
    latencies: list[float] | None,
) -> None:
    """Add a latency summary section to the metrics report.

    Computes count, mean, min, max, p50, p95, and p99 from the raw
    latency list. Skipped if the list is empty or None.

    Args:
        metrics: The :class:`Metrics` instance.
        section_id: Unique section identifier.
        section_title: Human-readable section title.
        latencies: Raw latency values in milliseconds.
    """
    if not latencies:
        return

    sorted_lat = sorted(latencies)
    count = len(sorted_lat)
    mean = sum(sorted_lat) / count
    p50_idx = max(0, math.ceil(count * 0.50) - 1)
    p95_idx = max(0, math.ceil(count * 0.95) - 1)
    p99_idx = max(0, math.ceil(count * 0.99) - 1)

    section = metrics.add_section(section_id, section_title)
    section.add(f"{section_id}_count", "count", count)
    section.add(f"{section_id}_mean", "mean", round(mean, 3))
    section.add(f"{section_id}_min", "min", round(sorted_lat[0], 3))
    section.add(f"{section_id}_max", "max", round(sorted_lat[-1], 3))
    section.add(f"{section_id}_p50", "p50", round(sorted_lat[p50_idx], 3))
    section.add(f"{section_id}_p95", "p95", round(sorted_lat[p95_idx], 3))
    section.add(f"{section_id}_p99", "p99", round(sorted_lat[p99_idx], 3))
