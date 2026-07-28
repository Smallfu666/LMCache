# SPDX-License-Identifier: Apache-2.0
"""Run-health ledger and per-worker statistics for the concurrent runner.

The bench must never report confident-but-wrong numbers, so every run tracks
two questions on a single monotonic ledger, :class:`RunHealth`:

- **Is this run's measurement trustworthy?** (``valid``)
- **Is the dedicated server safe to reuse afterwards?** (``server_reuse_safe``)

Severity can only escalate (``OK -> INVALID -> TAINTED``), never recover, so
no phase transition or cleanup-success path can silently downgrade a verdict
once raised:

- ``OK``      -- measurements valid; server reusable.
- ``INVALID`` -- measurements invalid; server still confirmably clean.
- ``TAINTED`` -- measurements invalid AND the server's clean state cannot be
  confirmed (a leaked prefetch job, held read locks, an unacknowledged
  cleanup), so the dedicated server must be restarted before reuse.

**Key invariant:** ``TAINTED`` implies ``INVALID``. Any condition that makes
server reuse unsafe also invalidates the run -- when teardown cannot confirm
clean state, that run's numbers are not published either. The combination
"measurements valid but server unsafe" (``valid=True`` with
``server_reuse_safe=False``) is therefore deliberately unrepresentable, which
is exactly what lets the two public booleans derive from one ledger value.

The per-request result and the single-request lifecycle live in
``helpers.py`` (:class:`~.helpers.RequestResult`, ``_process_request``); this
module only owns the health ledger, the per-worker counters, and their
deterministic merge. Each worker fills its own :class:`WorkerStats` and
:class:`RunVerdict`; the orchestrator thread merges them after all workers
join, so no verdict or stats object is ever shared across threads.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.bench.server_bench.helpers import RequestResult


class RunHealth(IntEnum):
    """Monotonic run-health severity; higher is worse, escalation only."""

    OK = 0
    INVALID = 1
    TAINTED = 2


@dataclass
class RunVerdict:
    """One worker's (or the merged) run-health verdict with its reasons.

    Escalated only through :meth:`invalidate` / :meth:`mark_server_unsafe`,
    which route through :meth:`_promote` so severity never decreases -- a
    later cleanup success cannot restore a verdict a failing step already
    raised. ``reasons`` keeps a stable order: the first is the primary
    failure, later entries are cleanup / taint context (e.g. "LOOKUP reply
    timed out" followed by "END_SESSION cleanup also timed out"), which is
    more useful for debugging than a bare "server tainted".

    Attributes:
        health: Current severity on the :class:`RunHealth` ledger.
        reasons: Ordered escalation reasons (primary first).
    """

    health: RunHealth = RunHealth.OK
    reasons: list[str] = field(default_factory=list)

    def _promote(self, health: RunHealth, reason: str) -> None:
        """Raise severity to at most *health* and append *reason*.

        Args:
            health: The severity floor to escalate to; ignored if the
                current severity is already higher.
            reason: Human-readable cause, always recorded in order.
        """
        if health > self.health:
            self.health = health
        self.reasons.append(reason)

    def invalidate(self, reason: str) -> None:
        """Escalate to at least ``INVALID`` and record *reason*.

        Args:
            reason: Why the run's numbers cannot be trusted.
        """
        self._promote(RunHealth.INVALID, reason)

    def mark_server_unsafe(self, reason: str) -> None:
        """Escalate to ``TAINTED`` (implies invalid) and record *reason*.

        Args:
            reason: Why the dedicated server may hold indeterminate state
                and must be restarted before reuse.
        """
        self._promote(RunHealth.TAINTED, reason)

    @property
    def valid(self) -> bool:
        """True only when the run's numbers are publishable."""
        return self.health is RunHealth.OK

    @property
    def server_reuse_safe(self) -> bool:
        """True unless the server may hold indeterminate state."""
        return self.health is not RunHealth.TAINTED


def merge_verdicts(per_worker: list[RunVerdict]) -> RunVerdict:
    """Merge worker-local verdicts on the orchestrator thread.

    The merged severity is the worst any worker reached; the merged reasons
    concatenate each worker's reasons in worker order (primary failures
    first within each worker). Workers never share a verdict, so this is the
    single point where their outcomes combine -- no locking required.

    Args:
        per_worker: One :class:`RunVerdict` per worker (may be empty).

    Returns:
        A merged :class:`RunVerdict`.
    """
    merged = RunVerdict()
    for v in per_worker:
        if v.health > merged.health:
            merged.health = v.health
        merged.reasons.extend(v.reasons)
    return merged


@dataclass
class WorkerStats:
    """Per-worker latency samples and outcome counters.

    Every worker fills its own instance (never shared), so no locking is
    needed; :func:`merge_worker_stats` combines them after all workers join.
    ``*_ms`` lists hold per-operation latencies in milliseconds. ``error``
    is a non-empty ``"<type>: <msg>"`` string if the worker aborted.

    Attributes:
        worker_id: Zero-based worker index; ``-1`` marks a merged result.
    """

    worker_id: int
    lookup_ms: list[float] = field(default_factory=list)
    retrieve_ms: list[float] = field(default_factory=list)
    store_ms: list[float] = field(default_factory=list)
    total_requests: int = 0
    checksum_ok: int = 0
    checksum_fail: int = 0
    error: str = ""


def account_result(stats: WorkerStats, result: "RequestResult") -> None:
    """Fold one successful request's latencies into *stats* in place.

    Appends the LOOKUP / STORE / RETRIEVE latency samples that the request
    populated (a full hit records no store latency; a full miss records no
    retrieve latency). Run-health (failure / taint) is a verdict concern on
    the worker's :class:`RunVerdict`, and the per-operation success / failure
    counters are deferred to the throughput follow-up, so neither is handled
    here -- this function only accumulates latency measurements.

    Args:
        stats: The worker's stats, mutated in place.
        result: The completed request result (from ``helpers``).
    """
    if result.lookup_ms is not None:
        stats.lookup_ms.append(result.lookup_ms)
    if result.store_ms is not None:
        stats.store_ms.append(result.store_ms)
    if result.retrieve_ms is not None:
        stats.retrieve_ms.append(result.retrieve_ms)


def merge_worker_stats(per_worker: list[WorkerStats]) -> WorkerStats:
    """Combine per-worker stats into one merged ``worker_id == -1`` result.

    Concatenates the latency sample lists and sums the counters. The first
    non-empty worker ``error`` becomes the merged error. Run-health is merged
    separately via :func:`merge_verdicts`; this function only aggregates
    measurements.

    Args:
        per_worker: One :class:`WorkerStats` per worker (may be empty).

    Returns:
        A merged :class:`WorkerStats` with ``worker_id == -1``.
    """
    merged = WorkerStats(worker_id=-1)
    for s in per_worker:
        merged.lookup_ms.extend(s.lookup_ms)
        merged.retrieve_ms.extend(s.retrieve_ms)
        merged.store_ms.extend(s.store_ms)
        merged.total_requests += s.total_requests
        merged.checksum_ok += s.checksum_ok
        merged.checksum_fail += s.checksum_fail
        if s.error and not merged.error:
            merged.error = s.error
    return merged
