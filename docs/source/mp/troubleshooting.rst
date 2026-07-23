.. _mp-troubleshooting:

Troubleshooting
===============

This page helps you diagnose problems when running LMCache in **multiprocess
(MP) mode** with a serving engine such as vLLM. It is organized by *symptom*:
each entry describes what you observe, the likely mechanism behind it, how to
confirm it, and what to try.

For how to *set up* observability (metrics, logs, traces), see
:doc:`observability/index`. For the trace record/replay tooling, see
:doc:`tracing_and_debugging`.

.. contents::
   :local:
   :depth: 1

First: localize the failure
---------------------------

MP mode has several moving parts -- the vLLM-side connector, the ZMQ transport,
the standalone ``lmcache server``, and any L2 storage backend. Before chasing a
specific fix, narrow down *where* the problem is:

- **Check the server is healthy.** The ``lmcache server`` HTTP frontend serves
  ``/metrics`` (Prometheus) and ``/status`` on ``--http-port`` (default
  ``8080``). If ``/status`` does not respond, the server itself is wedged or
  down -- look at its logs first.
- **Check the server logs.** Most store/retrieve and transport failures are
  logged on the server side, often with more detail than the engine sees.
- **Try reproducing with LMCache detached.** If the same symptom (a hang, a
  crash, low throughput) still occurs when you run vLLM *without* the LMCache
  connector, the root cause is elsewhere in the stack and not LMCache.
- **Confirm which connector implementation loaded** (see
  :ref:`mp-ts-low-hit-rate` below) -- a surprising number of "LMCache" issues
  are actually a version mismatch in the connector that vLLM resolved.

vLLM engine ``RPCTimeout`` / scheduler appears to stall under load
------------------------------------------------------------------

**Symptom.** Under sustained or bursty concurrent load, vLLM raises an engine
RPC timeout (or the scheduler appears frozen) some time after the LMCache
server is attached.

**Mechanism.** The connector issues requests to the ``lmcache server`` and
waits for the reply up to ``lmcache.mp.mq_timeout`` (default **300 s**). If the
server cannot service a request in time -- a stalled request handler, a
saturated server thread pool, or a blocked L2 backend -- that wait ties up the
caller and can surface on the vLLM side as an engine RPC timeout. Note the
timeout you see in vLLM is a *symptom*: the stall is usually downstream, in the
server or a storage tier.

**How to check.**

- Look for timeout errors in the ``lmcache server`` logs.
- Check ``/status`` on the HTTP frontend for queue depths and storage health.
- Reproduce with LMCache detached: if the stall persists without LMCache
  attached, it is not an LMCache problem (e.g. an attention/sampling kernel or
  driver issue in the base stack).

**What to try.**

- Raising ``lmcache.mp.mq_timeout`` is a *diagnostic*, not a fix -- it only
  widens the window; use it to confirm the stall is a slow reply rather than a
  hard hang.
- If the server thread pools are saturated, size them for your workload:
  ``--max-gpu-workers`` controls the pool that handles STORE/RETRIEVE
  (device-side transfers), and ``--max-cpu-workers`` controls the pool that
  handles LOOKUP and other CPU-side requests. A very small
  ``--max-gpu-workers`` serializes device transfers.
- If a storage backend is the bottleneck, see
  :ref:`mp-ts-miss-or-stuck-prefetch`.

MP mode fails to start / connector errors on a specific vLLM version
--------------------------------------------------------------------

**Symptom.** The worker or server fails during startup, or the connector raises
an import/attribute/version error, often after upgrading vLLM.

**Mechanism.** LMCache ships **versioned MP connector variants** matched to
vLLM's connector interface (which changes between releases). If the vLLM
version and the connector variant do not line up, startup can fail.

**How to check.**

- Note your exact vLLM version and confirm which connector variant is being
  loaded.
- Cross-check against the supported pairing in :doc:`configuration` and the
  :doc:`../getting_started/quickstart`.

**What to try.**

- Align vLLM and LMCache to a supported combination rather than mixing an old
  vLLM with a newer connector (or vice versa).

.. _mp-ts-low-hit-rate:

Warm cache hit rate is unexpectedly low / "cache never hits"
------------------------------------------------------------

**Symptom.** After a cold run that should have populated the cache, a warm run
with the same prefixes gets few or no hits.

**Mechanism.** Common causes:

- **Connector version skew.** vLLM may resolve ``LMCacheMPConnector`` to a copy
  bundled inside the vLLM tree rather than the connector from your installed
  LMCache. The bundled copy can be older and behave differently (for example,
  storing fewer chunks), so the hit rate drops even though your LMCache install
  is correct.
- **Namespace mismatch.** A differing ``cache_salt`` (or other key-scoping
  config) between the cold and warm runs produces different keys.
- **Server/client config mismatch** (chunk size, etc.).

**How to check.**

- Confirm the *module path* of the connector vLLM actually imported -- verify it
  points at your LMCache install, not the vLLM-bundled copy.
- Confirm the server is actually storing: the ``lmcache_mp_*`` series on
  ``/metrics`` should grow during the cold run (metrics are lazy -- drive some
  traffic first).

**What to try.**

- Force the connector from your LMCache install (e.g. via
  ``kv_connector_module_path``) so the bundled copy is not used.
- Ensure the cold and warm runs use the same key-scoping config.

.. _mp-ts-miss-or-stuck-prefetch:

Lookups miss although data was stored / prefetch appears stuck
--------------------------------------------------------------

**Symptom.** Keys that should be warm return misses, or retrieval from an L2
tier never completes.

**Mechanism.** On a lookup miss in L1, the server prefetches the missing chunks
from L2 in the background and the client polls for completion. A slow or
unresponsive L2 backend (filesystem, S3-compatible object store, etc.) can
delay or stall that prefetch, so the data never becomes available to the
engine.

.. VERIFY: confirm current dev's exact prefetch-timeout behavior per L2 adapter
   before asserting anything stronger than "slow backend delays/stalls prefetch".

**How to check.**

- Check ``/status`` for prefetch and storage-tier health.
- Test the L2 backend independently. For S3-compatible stores (including
  MinIO), verify the endpoint, credentials, and addressing style -- see
  :doc:`l2_storage/index`.

**What to try.**

- Fix or replace the unhealthy backend; confirm connectivity from the server's
  environment (not just your workstation).

Collecting diagnostics
----------------------

- Enable and read metrics/logs/traces as described in
  :doc:`observability/index`.
- Use ``lmcache trace record`` / ``lmcache trace replay`` to capture and reproduce
  a request timeline offline -- see :doc:`tracing_and_debugging`.
- When filing an issue, include: LMCache and vLLM versions, the connector
  variant, the ``lmcache server`` command line, the server logs around the
  failure, and whether the problem reproduces with LMCache detached.
