# SPDX-License-Identifier: Apache-2.0
"""Concurrent orchestration for the ``lmcache bench server`` pair workload.

This module owns *only* the concurrency layer around the existing single
request path (:func:`~.helpers._process_request`): one worker per requested
degree of concurrency, each with its own MP client, its own registered
KV-cache context, its own non-overlapping block band, a two-phase start
barrier so registration/allocation is excluded from the measured window, and
a per-worker :class:`~.result.RunVerdict` that the orchestrator thread merges
after every worker joins. The same path runs at ``concurrency == 1``.

The runner never issues an RPC itself. The client, allocation, REGISTER,
per-request flow, UNREGISTER, and resource close are injected as a
:class:`WorkerRuntime` (the real implementation lives in ``command.py``; tests
pass fakes). The runner's job is the *contract*: keep worker state private,
gate the measured phase, translate each request outcome and each teardown
outcome into the worker's monotonic verdict, and never let a failure produce a
confident-but-wrong result.

Health mapping (see :class:`~.result.RunHealth`):

- a request that returns ``failure`` -> ``verdict.invalidate`` (run invalid);
- a request that returns ``server_tainted``, or raises ``ServerTaintedError``,
  or an UNREGISTER that cannot be acknowledged -> ``verdict.mark_server_unsafe``
  (run invalid AND server must be restarted);
- a ``KeyboardInterrupt`` / ``ServerTaintedInterrupt`` -> the run is marked
  interrupted (exit 130 is the caller's concern) and invalid.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable
import secrets
import threading

# First Party
from lmcache.cli.commands.bench.server_bench.helpers import (
    ServerTaintedError,
    ServerTaintedInterrupt,
)
from lmcache.cli.commands.bench.server_bench.result import (
    RunVerdict,
    WorkerStats,
    account_result,
    merge_verdicts,
    merge_worker_stats,
)
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.bench.server_bench.helpers import RequestResult
    from lmcache.v1.multiprocess.mq import MessageQueueClient

logger = init_logger(__name__)

# Upper bound on how long the orchestrator (and each worker) waits at the
# start barrier before treating the phase as broken, so a worker that dies
# during setup never parks its siblings forever.
_START_BARRIER_TIMEOUT_S = 120.0


@dataclass
class WorkerContext:
    """Per-worker state; owned by exactly one worker, never shared.

    Attributes:
        worker_id: Zero-based worker index.
        instance_id: The registered KV-cache context id this worker owns
            (a per-run nonce base + ``worker_id``, so ids stay distinct
            within a run and practically distinct across runs).
        band_base: First block of this worker's band in the shared block
            space.
        band_blocks: Number of blocks in this worker's band; its requests
            stay within ``[band_base, band_base + band_blocks)``.
        client: This worker's MP client (distinct DEALER identity), set by
            :meth:`WorkerRuntime.setup`.
        registered_maybe: ``True`` once REGISTER has been *submitted* for
            this worker (set before the ack is observed), so teardown
            best-effort UNREGISTERs it even after a timeout / interrupt.
        verdict: This worker's monotonic run-health verdict.
        stats: This worker's latency / counter samples.
        keepalive: Opaque handle the runtime uses to keep tensors / SHM
            mappings alive for the worker's lifetime (closed in teardown).
    """

    worker_id: int
    instance_id: int
    band_base: int
    band_blocks: int
    client: "MessageQueueClient | None" = None
    registered_maybe: bool = False
    verdict: RunVerdict = field(default_factory=RunVerdict)
    stats: WorkerStats = field(init=False)
    keepalive: object = None

    def __post_init__(self) -> None:
        self.stats = WorkerStats(worker_id=self.worker_id)


@dataclass
class WorkerRuntime:
    """The injected primitives the runner orchestrates.

    The real implementation (client creation, allocation, RPCs) lives in
    ``command.py``; tests pass fakes. The runner supplies the barrier,
    verdict, and teardown *order* -- these callables do the actual work.

    Attributes:
        setup: Create the worker's client, allocate its band-size KV cache,
            set ``registered_maybe = True``, and submit REGISTER. Mutates the
            :class:`WorkerContext` in place and raises on failure.
        process_request: Run one pair pass for ``(context, seq_no,
            pass_label)`` and return its :class:`~.helpers.RequestResult`
            (or ``None`` for a legal sub-chunk skip). May raise
            ``ServerTaintedError`` / ``ServerTaintedInterrupt``.
        unregister: Best-effort UNREGISTER of the worker's context; returns
            ``True`` if the server acknowledged, ``False`` on timeout. May
            raise on a transport error.
        close: Release the worker's client / SHM / tensor resources. Must not
            raise for an ordinary close failure (logged, not fatal).
    """

    setup: Callable[["WorkerContext"], None]
    process_request: Callable[
        ["WorkerContext", int, str], "RequestResult | None"
    ]
    unregister: Callable[["WorkerContext"], bool]
    close: Callable[["WorkerContext"], None]


@dataclass
class RunnerResult:
    """The merged outcome of a concurrent run.

    Attributes:
        verdict: The merged run-health verdict (worst worker wins).
        stats: The merged per-worker statistics.
        interrupted: ``True`` if a ``KeyboardInterrupt`` reached any worker or
            the orchestrator; the caller maps this to exit 130.
    """

    verdict: RunVerdict
    stats: WorkerStats
    interrupted: bool = False


def _noop() -> None:
    """Default measurement hook: do nothing."""


def _await_barrier(barrier: threading.Barrier) -> bool:
    """Wait on *barrier*; return ``False`` if it was broken/aborted.

    Args:
        barrier: The barrier to wait on (bounded by
            :data:`_START_BARRIER_TIMEOUT_S` so a dead party never parks the
            rest forever).

    Returns:
        ``True`` if all parties arrived, ``False`` on
        :class:`threading.BrokenBarrierError` (a party failed or timed out).
    """
    try:
        barrier.wait(timeout=_START_BARRIER_TIMEOUT_S)
        return True
    except threading.BrokenBarrierError:
        return False


def _worker_seq_numbers(
    worker_id: int,
    concurrency: int,
    start: int,
    end: "int | None",
    stop_event: threading.Event,
) -> Iterator[int]:
    """Round-robin-partition the ``[start, end)`` range across workers.

    Worker ``w`` yields ``start + w, start + w + N, start + w + 2N, ...``
    (``N`` = concurrency), so the workers together cover the sequence range
    with no two ever sharing a ``seq_no`` -- and therefore no shared
    ``request_id`` / cache-key namespace. The bound follows the existing CLI
    contract:

    - ``end`` set -> a finite run, yielding while ``seq < end``;
    - ``end is None`` -> the legacy infinite run, yielding forever until
      *stop_event* is set (a sibling failure) or the run is interrupted.

    Args:
        worker_id: This worker's index.
        concurrency: Total worker count (the round-robin stride).
        start: Inclusive base sequence number.
        end: Exclusive upper bound, or ``None`` for an infinite run.
        stop_event: Shared cancellation flag; ends the generator early.

    Yields:
        This worker's sequence numbers in increasing order.
    """
    seq = start + worker_id
    while (end is None or seq < end) and not stop_event.is_set():
        yield seq
        seq += concurrency


def _accept_pass(
    wc: "WorkerContext",
    result: "RequestResult | None",
    seq_no: int,
    pass_label: str,
    stop_event: threading.Event,
) -> bool:
    """Fold one pass's outcome into the worker's verdict; return keep-going.

    A pair pass that transfers nothing (``None``), a tainting result, or a
    plain failure each stops the worker (fail-close) after recording the
    reason. Only a clean pass returns ``True``.

    Args:
        wc: The worker context (verdict / stats mutated in place).
        result: The pass result from :attr:`WorkerRuntime.process_request`.
        seq_no: The sequence number of this pass.
        pass_label: ``"cold"`` or ``"warm"``.
        stop_event: Shared cancellation flag, set on any failure.

    Returns:
        ``True`` if the pass succeeded and the worker should continue.
    """
    if result is None:
        wc.verdict.invalidate(
            "[w%d seq %d %s] request transferred nothing (fewer tokens than "
            "one chunk); an empty pair pass is invalid"
            % (wc.worker_id, seq_no, pass_label)
        )
        stop_event.set()
        return False
    account_result(wc.stats, result)
    if result.server_tainted:
        wc.verdict.mark_server_unsafe(
            "[w%d seq %d %s] %s"
            % (wc.worker_id, seq_no, pass_label, result.failure or "server tainted")
        )
        stop_event.set()
        return False
    if result.failure:
        wc.verdict.invalidate(
            "[w%d seq %d %s] %s" % (wc.worker_id, seq_no, pass_label, result.failure)
        )
        stop_event.set()
        return False
    return True


def _verify_pair_checksum(
    wc: "WorkerContext",
    cold: "RequestResult",
    warm: "RequestResult",
    seq_no: int,
    stop_event: threading.Event,
) -> bool:
    """Fail-close checksum oracle for one pair; return keep-going.

    The cold pass just stored this sequence, so the warm pass must hit every
    chunk and reproduce the cold digests exactly. A short hit, a missing or
    length-mismatched digest, or any unequal element invalidates the run --
    it is never a silently skipped comparison.

    Args:
        wc: The worker context (verdict / stats mutated in place).
        cold: The cold (STORE) pass result.
        warm: The warm (RETRIEVE) pass result.
        seq_no: The sequence number of this pair.
        stop_event: Shared cancellation flag, set on a mismatch.

    Returns:
        ``True`` if the digests matched and the worker should continue.
    """
    expected = warm.total_chunks
    if warm.hit_chunks < expected:
        wc.stats.checksum_fail += 1
        wc.verdict.invalidate(
            "[w%d seq %d] checksum verification failed (fail-close): the warm "
            "pass hit %d of %d chunks, so the cold STORE was not fully "
            "retrievable and nothing could be verified"
            % (wc.worker_id, seq_no, warm.hit_chunks, expected)
        )
        stop_event.set()
        return False
    cold_checksums = cold.checksums
    warm_checksums = warm.checksums
    verified = (
        cold_checksums is not None
        and warm_checksums is not None
        and len(cold_checksums) == expected
        and len(warm_checksums) == expected
        and cold_checksums == warm_checksums
    )
    if not verified:
        wc.stats.checksum_fail += 1
        wc.verdict.invalidate(
            "[w%d seq %d] checksum verification failed (fail-close): the warm "
            "RETRIEVE did not reproduce the cold STORE digests"
            % (wc.worker_id, seq_no)
        )
        stop_event.set()
        return False
    wc.stats.checksum_ok += 1
    return True


def _drive_pairs(
    wc: "WorkerContext",
    runtime: WorkerRuntime,
    seq_numbers: Iterable[int],
    checksum_on: bool,
    stop_event: threading.Event,
) -> None:
    """Run the pair workload (cold STORE, warm RETRIEVE, verify) for a worker.

    Stops at the first failing pass or checksum mismatch (fail-close) or when
    ``stop_event`` is set by another worker. A ``ServerTaintedError`` /
    ``ServerTaintedInterrupt`` raised by a pass propagates to the caller.

    Args:
        wc: This worker's context.
        runtime: The injected worker primitives.
        seq_numbers: This worker's sequence numbers.
        checksum_on: Whether to run the checksum oracle per pair.
        stop_event: Shared cancellation flag.
    """
    for seq_no in seq_numbers:
        if stop_event.is_set():
            return
        cold = runtime.process_request(wc, seq_no, "cold")
        if not _accept_pass(wc, cold, seq_no, "cold", stop_event):
            return
        warm = runtime.process_request(wc, seq_no, "warm")
        if not _accept_pass(wc, warm, seq_no, "warm", stop_event):
            return
        wc.stats.total_requests += 1
        # _accept_pass returns False for a None result, so both are non-None
        # here; this narrows the type for the checksum oracle (never hit).
        if cold is None or warm is None:
            return
        if checksum_on and not _verify_pair_checksum(
            wc, cold, warm, seq_no, stop_event
        ):
            return


def _teardown_worker(wc: "WorkerContext", runtime: WorkerRuntime) -> None:
    """Best-effort teardown; escalate the verdict when cleanup is unconfirmed.

    Order: UNREGISTER the (possibly) registered context, then release
    client / SHM resources. UNREGISTER of a context the server never created
    is a safe no-op, so it is always attempted once ``registered_maybe`` is
    set. A cleanup that cannot be confirmed (a timeout, or a raised
    UNREGISTER) marks the server unsafe; a confirmed UNREGISTER never lowers
    an already-raised verdict (:meth:`RunVerdict` only escalates).

    Args:
        wc: The worker context.
        runtime: The injected worker primitives.
    """
    if wc.registered_maybe:
        try:
            acknowledged = runtime.unregister(wc)
        except BaseException as exc:  # noqa: BLE001 - re-cast as a taint
            wc.verdict.mark_server_unsafe(
                "worker %d UNREGISTER raised (%s: %s); server may still hold "
                "the context, restart required"
                % (wc.worker_id, type(exc).__name__, exc)
            )
        else:
            if not acknowledged:
                wc.verdict.mark_server_unsafe(
                    "worker %d UNREGISTER timed out; server may still hold the "
                    "context, restart required" % wc.worker_id
                )
    try:
        runtime.close(wc)
    except Exception as exc:  # noqa: BLE001 - close failures are not fatal
        logger.warning("worker %d resource close failed: %s", wc.worker_id, exc)


def _run_worker(
    wc: "WorkerContext",
    runtime: WorkerRuntime,
    seq_numbers: Iterable[int],
    checksum_on: bool,
    ready_barrier: threading.Barrier,
    start_event: threading.Event,
    done_barrier: threading.Barrier,
    teardown_event: threading.Event,
    stop_event: threading.Event,
    interrupted: threading.Event,
) -> None:
    """One worker: setup -> ready -> measured workload -> stop -> teardown.

    Two barriers bracket the measured window so the orchestrator can time
    exactly the workload: ``ready_barrier`` after setup (setup / REGISTER
    excluded), ``done_barrier`` after the workload. The worker then waits on
    ``teardown_event`` so teardown / UNREGISTER runs only after the
    orchestrator has stopped measurement -- and always on this worker's own
    thread, in the ``finally``. Any failure aborts the barriers so no party
    parks forever.

    Args:
        wc: This worker's context.
        runtime: The injected worker primitives.
        seq_numbers: This worker's sequence numbers.
        checksum_on: Whether to run the checksum oracle per pair.
        ready_barrier: ``concurrency + 1`` party barrier after setup.
        start_event: Released by the orchestrator once measurement started.
        done_barrier: ``concurrency + 1`` party barrier after the workload.
        teardown_event: Released by the orchestrator after measurement stops.
        stop_event: Shared cooperative cancellation flag.
        interrupted: Set when a ``KeyboardInterrupt`` reaches this worker.
    """
    try:
        try:
            runtime.setup(wc)
        except BaseException as exc:  # noqa: BLE001 - surface as verdict, not crash
            if isinstance(exc, KeyboardInterrupt):
                interrupted.set()
                wc.verdict.mark_server_unsafe(
                    "worker %d interrupted during setup" % wc.worker_id
                )
            else:
                wc.verdict.invalidate(
                    "worker %d setup failed (%s: %s)"
                    % (wc.worker_id, type(exc).__name__, exc)
                )
            stop_event.set()
            # Wake the orchestrator and siblings parked on either barrier.
            ready_barrier.abort()
            done_barrier.abort()
            return

        if not _await_barrier(ready_barrier):
            # Another worker failed setup (or timed out): skip the workload.
            return
        start_event.wait()
        if not stop_event.is_set():
            _drive_pairs(wc, runtime, seq_numbers, checksum_on, stop_event)
        # Sync after the workload so the orchestrator can stop measurement
        # before any worker starts tearing down, then wait for its release.
        if not _await_barrier(done_barrier):
            return
        teardown_event.wait()
    except ServerTaintedInterrupt as exc:
        interrupted.set()
        wc.verdict.mark_server_unsafe("worker %d: %s" % (wc.worker_id, exc))
        stop_event.set()
        done_barrier.abort()
    except KeyboardInterrupt:
        interrupted.set()
        wc.verdict.invalidate("worker %d interrupted" % wc.worker_id)
        stop_event.set()
        done_barrier.abort()
    except ServerTaintedError as exc:
        wc.verdict.mark_server_unsafe("worker %d: %s" % (wc.worker_id, exc))
        stop_event.set()
        done_barrier.abort()
    except Exception as exc:  # noqa: BLE001 - surface as invalid, not a crash
        wc.verdict.invalidate(
            "worker %d: %s: %s" % (wc.worker_id, type(exc).__name__, exc)
        )
        stop_event.set()
        done_barrier.abort()
    finally:
        _teardown_worker(wc, runtime)


def run_concurrent_pairs(
    runtime: WorkerRuntime,
    *,
    concurrency: int,
    total_blocks: int,
    request_blocks: int,
    checksum_on: bool = True,
    start: int = 0,
    end: "int | None" = None,
    on_measure_start: Callable[[], None] = _noop,
    on_measure_stop: Callable[[], None] = _noop,
) -> RunnerResult:
    """Run the pair workload across *concurrency* workers and merge the result.

    Each worker owns one client, one registered context (id ``nonce +
    worker_id``), one non-overlapping block band
    (``total_blocks // concurrency`` blocks starting at
    ``worker_id * band_blocks``), and a globally unique, round-robin slice of
    the ``[start, end)`` sequence range (``end is None`` = the legacy infinite
    run, stopped only by a failure or interrupt). Two barriers bracket the
    measured workload so the measurement hooks
    time exactly the workload: ``on_measure_start`` fires on this
    (orchestrator) thread once every worker has finished setup / REGISTER, and
    ``on_measure_stop`` fires once every worker has finished the workload and
    before any worker tears down. So setup / REGISTER and teardown / UNREGISTER
    are excluded from the measured window and the workload is included.
    ``concurrency == 1`` takes the same path.

    Args:
        runtime: The injected worker primitives (real in ``command.py``,
            fakes in tests).
        concurrency: Number of concurrent workers (>= 1).
        total_blocks: Total paged blocks shared across all bands.
        request_blocks: Blocks a single request needs; the run is refused if
            a band cannot hold one request.
        checksum_on: Whether each pair runs the checksum oracle.
        start: Inclusive base sequence number.
        end: Exclusive upper bound of the shared sequence range, or ``None``
            for the legacy infinite run (stopped only by a failure /
            interrupt).
        on_measure_start: Called on the orchestrator thread when the measured
            window opens (all workers set up, before the first request).
        on_measure_stop: Called on the orchestrator thread when the measured
            window closes (all workers done, before any teardown).

    Returns:
        The merged :class:`RunnerResult`.

    Raises:
        ValueError: If ``concurrency < 1`` or a band cannot hold one request.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1, got %d" % concurrency)
    band_blocks = total_blocks // concurrency
    if band_blocks < request_blocks:
        raise ValueError(
            "each worker's band is %d blocks (total_blocks %d // concurrency "
            "%d) but one request needs %d; reduce --concurrency or the request "
            "size" % (band_blocks, total_blocks, concurrency, request_blocks)
        )

    # A per-run nonce keeps context ids (and, via the request path, request
    # ids) from colliding with a prior run's still-registered state.
    instance_base = secrets.randbits(62) + 1
    contexts = [
        WorkerContext(
            worker_id=w,
            instance_id=instance_base + w,
            band_base=w * band_blocks,
            band_blocks=band_blocks,
        )
        for w in range(concurrency)
    ]

    ready_barrier = threading.Barrier(concurrency + 1)
    done_barrier = threading.Barrier(concurrency + 1)
    start_event = threading.Event()
    teardown_event = threading.Event()
    stop_event = threading.Event()
    interrupted = threading.Event()

    threads = [
        threading.Thread(
            target=_run_worker,
            args=(
                wc,
                runtime,
                _worker_seq_numbers(wc.worker_id, concurrency, start, end, stop_event),
                checksum_on,
                ready_barrier,
                start_event,
                done_barrier,
                teardown_event,
                stop_event,
                interrupted,
            ),
            name="bench-worker-%d" % wc.worker_id,
        )
        for wc in contexts
    ]
    for t in threads:
        t.start()

    measuring = False
    hook_error = ""
    try:
        if _await_barrier(ready_barrier):
            # Every worker finished setup: open the measured window, then
            # release the workers to run the workload concurrently. A hook
            # that raises (not a Ctrl-C) is not fatal: the run is invalidated
            # and the workers are still released so teardown runs.
            try:
                on_measure_start()
                measuring = True
            except Exception as exc:  # noqa: BLE001 - invalid run, not a crash
                hook_error = "measurement-start hook failed: %s: %s" % (
                    type(exc).__name__,
                    exc,
                )
                stop_event.set()
                done_barrier.abort()
        else:
            # A worker failed setup; the workload never starts.
            stop_event.set()
        start_event.set()
        # Wait for every worker to finish the workload (or abort), then close
        # the measured window before releasing anyone to tear down.
        _await_barrier(done_barrier)
        if measuring:
            try:
                on_measure_stop()
            except Exception as exc:  # noqa: BLE001 - invalid run, not a crash
                hook_error = "measurement-stop hook failed: %s: %s" % (
                    type(exc).__name__,
                    exc,
                )
            measuring = False
        # Release every worker (parked on teardown_event) so teardown runs on
        # their own threads even after a hook failure.
        teardown_event.set()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        # Ctrl-C on the orchestrator thread while it waits: stop, abort both
        # barriers and release both gates so every parked worker wakes and
        # tears down, close measurement if it was open, then join and report.
        interrupted.set()
        stop_event.set()
        ready_barrier.abort()
        done_barrier.abort()
        start_event.set()
        teardown_event.set()
        if measuring:
            on_measure_stop()
        for t in threads:
            t.join()

    verdict = merge_verdicts([wc.verdict for wc in contexts])
    stats = merge_worker_stats([wc.stats for wc in contexts])
    if hook_error:
        verdict.invalidate(hook_error)
    if interrupted.is_set():
        verdict.invalidate("run interrupted (Ctrl-C)")
    return RunnerResult(verdict=verdict, stats=stats, interrupted=interrupted.is_set())
