.. _lmcache-bench:

lmcache bench
=============

The ``lmcache bench`` command runs sustained performance benchmarks. It has
three sub-commands, each targeting a different layer of the stack:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Sub-command
     - Description
   * - ``engine``
     - Benchmark an inference engine (e.g. vLLM) with workloads that
       exercise different KV-cache reuse patterns.
   * - ``server``
     - End-to-end sanity test against a running LMCache MP cache server
       (ZMQ + HTTP). Requires the full ``lmcache`` install and a GPU.
   * - ``l2``
     - Throughput / latency benchmark against an L2 cache adapter
       (store / lookup / load).

.. code-block:: bash

   lmcache bench {engine,server,l2} [options]


.. _lmcache-bench-engine:

engine
------

The ``lmcache bench engine`` command runs sustained performance benchmarks
against an inference engine (e.g., vLLM). It supports multiple workload types
that exercise different caching patterns and reports TTFT, decoding speed, and
throughput metrics.

.. code-block:: bash

   lmcache bench engine [options]

There are three ways to configure the benchmark:

1. **CLI arguments** -- pass all options on the command line.
2. **Interactive mode** -- run ``lmcache bench engine`` without required args
   and follow the step-by-step prompts.
3. **Config file** -- save a configuration to JSON and replay it with
   ``--config``.


Quick Start
~~~~~~~~~~~

**Minimal (with all required arguments):**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080

**Interactive mode (guided setup):**

.. code-block:: bash

   lmcache bench engine

The interactive mode walks you through each required setting, then asks
whether you want to configure general and workload-specific options or use
defaults.

**From a saved config file:**

.. code-block:: bash

   lmcache bench engine --engine-url http://localhost:8000 \
       --config my_bench.json

Config files contain benchmark parameters (workload, KV cache settings, etc.)
but not the engine URL, so you can reuse the same config against different
engines.

**Export a config without running the benchmark:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080 \
       --export-config my_bench.json

This resolves all auto-detected values (model name, tokens per GB) and saves
them to a portable JSON file that works without an LMCache server.

**Non-interactive mode (for scripts and CI):**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080 \
       --no-interactive

Errors immediately if any required argument is missing, instead of entering
interactive mode. Useful in automated pipelines.

If you don't have an LMCache server, you can pass ``--tokens-per-gb-kvcache``
directly instead of ``--lmcache-url``
(see :ref:`bench-tokens-per-gb` for how to find this value).


General Options
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 60

   * - Flag
     - Required
     - Description
   * - ``--config FILE``
     - No
     - Load configuration from a JSON file. Skips interactive mode.
       CLI flags override values in the file. The engine URL is not
       stored in config files and must be provided separately.
   * - ``--export-config FILE``
     - No
     - Export resolved configuration to a JSON file and exit. Does not
       run the benchmark. Auto-detected values (model, tokens per GB)
       are resolved and saved so the config is portable. Environment-
       specific values (engine URL, LMCache URL) are excluded.
   * - ``--no-interactive``
     - No
     - Disable interactive mode. Errors if required arguments are
       missing instead of prompting. Useful for scripts and CI.
   * - ``--engine-url URL``
     - Yes
     - Inference engine URL (e.g., ``http://localhost:8000``).
       Set ``OPENAI_API_KEY`` env var if authentication is needed.
   * - ``--workload TYPE``
     - Yes
     - Workload type: ``long-doc-qa``, ``multi-round-chat``,
       ``long-doc-permutator``, ``prefix-suffix-tuner``, or
       ``random-prefill``.
   * - ``--tokens-per-gb-kvcache N``
     - \*
     - Tokens per GB of KV cache. Required unless ``--lmcache-url`` is set.
       See :ref:`bench-tokens-per-gb` for how to find this value.
   * - ``--lmcache-url URL``
     - No
     - LMCache HTTP server URL. When provided, ``--tokens-per-gb-kvcache``
       is auto-detected from the server.
   * - ``--model NAME``
     - No
     - Model name. Auto-detected from the engine if omitted.
   * - ``--kv-cache-volume GB``
     - No
     - Target active KV cache volume in GB (default: 100).
   * - ``--seed N``
     - No
     - Random seed (default: 42).
   * - ``--output-dir DIR``
     - No
     - Directory for CSV and JSON output files (default: current directory).
   * - ``--no-csv``
     - No
     - Skip CSV export.
   * - ``--json``
     - No
     - Export a JSON summary file.
   * - ``-q`` / ``--quiet``
     - No
     - Suppress the real-time progress display.


.. _bench-tokens-per-gb:

Finding ``--tokens-per-gb-kvcache``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have an LMCache server running, the easiest approach is to pass
``--lmcache-url`` and let the tool auto-detect the value.

If you are using **vLLM without LMCache**, look for these lines in vLLM's
startup log:

.. code-block:: text

   INFO: Available KV cache memory: 12.34 GiB
   INFO: GPU KV cache size: 567,890 tokens

Then compute::

   tokens_per_gb = 567890 / 12.34 = 46,020


Workloads
~~~~~~~~~

long-doc-qa
^^^^^^^^^^^

Simulates repeated Q&A over long documents. Warmup sends each document once
to populate the KV cache, then benchmark queries are dispatched with
semaphore-controlled concurrency.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--ldqa-document-length``
     - 10000
     - Token length of each synthetic document.
   * - ``--ldqa-query-per-document``
     - 2
     - Number of questions asked per document.
   * - ``--ldqa-shuffle-policy``
     - random
     - Request ordering: ``random`` (shuffled) or ``tile`` (round-by-round).
   * - ``--ldqa-num-inflight-requests``
     - 3
     - Maximum concurrent in-flight requests.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-qa \
       --lmcache-url http://localhost:8080 \
       --kv-cache-volume 50 \
       --ldqa-document-length 8000 \
       --ldqa-query-per-document 4 \
       --ldqa-shuffle-policy tile


multi-round-chat
^^^^^^^^^^^^^^^^

Simulates multi-round chat with stateful sessions. Creates concurrent user
sessions, dispatches requests at a fixed QPS rate, and records responses in
session history so each subsequent query includes prior context.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--mrc-shared-prompt-length``
     - 2000
     - System prompt token length per session.
   * - ``--mrc-chat-history-length``
     - 10000
     - Pre-filled chat history token length.
   * - ``--mrc-user-input-length``
     - 50
     - Tokens per user query.
   * - ``--mrc-output-length``
     - 200
     - Max tokens to generate per response.
   * - ``--mrc-qps``
     - 1.0
     - Target queries per second.
   * - ``--mrc-duration``
     - 60.0
     - Benchmark duration in seconds.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload multi-round-chat \
       --lmcache-url http://localhost:8080 \
       --mrc-qps 2.0 \
       --mrc-duration 120


long-doc-permutator
^^^^^^^^^^^^^^^^^^^

Stress-tests blended KV cache reuse by sending permutations of a set of context
documents. Each request concatenates all context documents in a different order:

.. code-block:: text

   [System Prompt] + [Doc_i1] + [Doc_i2] + ... + [Doc_iN]

A single dummy warmup request is sent before the benchmark phase. Requests are
dispatched with semaphore-controlled concurrency.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--ldp-num-contexts``
     - 5
     - Number of unique context documents.
   * - ``--ldp-context-length``
     - 5000
     - Token length of each context document.
   * - ``--ldp-system-prompt-length``
     - 1000
     - Token length of the shared system prompt. Use ``0`` for no system prompt.
   * - ``--ldp-num-permutations``
     - 10
     - Number of distinct permutations to send. Capped at N! where
       N = ``--ldp-num-contexts``.
   * - ``--ldp-num-inflight-requests``
     - 1
     - Maximum concurrent in-flight requests.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload long-doc-permutator \
       --lmcache-url http://localhost:8080 \
       --ldp-num-contexts 4 \
       --ldp-context-length 8000 \
       --ldp-num-permutations 24 \
       --ldp-num-inflight-requests 2


prefix-suffix-tuner
^^^^^^^^^^^^^^^^^^^

A two-pass sequential workload designed to be run **unchanged** across
three LMCache configurations to demonstrate the value of each cache tier
(L0 HBM, L1 DRAM, L2 disk):

.. list-table::
   :header-rows: 1
   :widths: 15 25 30 30

   * - Baseline
     - LMCache config
     - Targeted overflow
     - Expected pass-2 hits
   * - 1
     - vanilla vLLM (L0 only)
     - L0 (HBM)
     - none -- every request a cold prefill
   * - 2
     - vLLM + LMCache L1 + L2
     - L1 (DRAM)
     - L2 prefix hits (suffix recomputed)
   * - 3
     - vLLM + LMCache L1 + L2 + CacheBlend
     - L1 (DRAM)
     - L2 prefix hits + CacheBlend suffix hits

Set ``--kv-cache-volume`` to the size in GB of the tier you want to overflow
(L0 size for Baseline 1, L1 size for Baselines 2 and 3). The workload itself
is identical across baselines.

Each request has the layout::

   [prefix_i with unique-ID][random breaker][shared suffix]

- ``num_prefixes`` distinct prefixes, each starting with ``PREFIX_<8-hex>``
  so the prefix's tokenized hash differs across the pool.
- A fresh random 32-token breaker per request, defeating ordinary prefix
  caching past the prefix boundary.
- A single shared suffix used by every request -- the only entry CacheBlend
  can reuse.

Pass 1 (warmup) sends each prefix once to populate the cache; its stats are
discarded. Pass 2 sends them again in identical order. Because LRU evicts
the next-needed prefix on each pass-2 access, even a 1.05x overflow of the
targeted tier is enough to make every pass-2 request miss that tier and
fall through to the next one.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--psf-context-length``
     - 8000
     - Total tokens per request (prefix + breaker + suffix).
   * - ``--psf-prefix-ratio``
     - 0.8
     - Fraction of context-length used by the prefix. Must be in (0.0, 1.0).
       The remainder (minus a 32-token breaker) is the shared suffix.
   * - ``--psf-thrash``
     - 20.0
     - **Size in GB of the KV-cache tier to overflow.** Use the L0 (HBM)
       size for vanilla vLLM, or the L1 (LMCache DRAM) size for tiered
       baselines. The workload sizes its prefix pool to slightly more than
       this (5% overflow internally), enough to drive every pass-2 request
       to a miss of that tier under sequential dispatch + LRU.

The number of pass-2 (measured) requests equals the prefix pool size,
computed as
``floor(psf_thrash * 1.05 * tokens_per_gb / prefix_tokens)``.
``--kv-cache-volume`` is unused by this workload — sizing is driven solely
by ``--psf-thrash``.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload prefix-suffix-tuner \
       --lmcache-url http://localhost:8080 \
       --psf-context-length 8000 \
       --psf-prefix-ratio 0.8 \
       --psf-thrash 100

.. note::
   For the analytical-model claim "thrash ≈ L1 size → ~0% LMCache hit rate"
   to hold empirically, the LMCache server must be started with
   ``--eviction-ratio 0.99`` (default ``0.20`` only clears 20% per cycle,
   leaving ~60% of pass-1 content in cache through pass 2):

   .. code-block:: bash

      lmcache server --l1-size-gb <SIZE> --eviction-policy LRU \
          --eviction-trigger-watermark 0.80 \
          --eviction-ratio 0.99

   The workload itself sleeps 5 seconds between pass 1 (warmup) and pass 2
   (measured), so LMCache's 1Hz batched-eviction polling thread has time
   to actually run.  Without that sleep, fast benchmarks complete before
   any eviction fires.


random-prefill
^^^^^^^^^^^^^^

Fires all requests simultaneously with ``max_tokens=1`` to measure pure
prefill performance. No warmup phase.

.. list-table::
   :header-rows: 1
   :widths: 35 10 55

   * - Flag
     - Default
     - Description
   * - ``--rp-request-length``
     - 10000
     - Token length per prefill request.
   * - ``--rp-num-requests``
     - 50
     - Number of requests to fire.

**Example:**

.. code-block:: bash

   lmcache bench engine \
       --engine-url http://localhost:8000 \
       --workload random-prefill \
       --lmcache-url http://localhost:8080 \
       --rp-request-length 15000 \
       --rp-num-requests 100


Interactive Mode
~~~~~~~~~~~~~~~~

.. image:: /_static/bench_interactive_demo.gif
   :alt: Interactive mode demo
   :width: 100%

When ``--engine-url`` or ``--workload`` is not provided (and
``--no-interactive`` is not set), the tool enters interactive mode. It guides
you through four phases:

1. **Required settings** -- engine URL, workload type, LMCache server
   (or tokens per GB).
2. **General settings** (optional gate) -- model name, KV cache volume.
3. **Workload settings** (optional gate) -- workload-specific parameters.
4. **Summary and action** -- review configuration, then start the benchmark
   or export to a JSON file.

Each prompt focuses on a single setting. Selection prompts use arrow keys;
text and number prompts accept typed input with defaults shown in brackets.

.. code-block:: text

   ══════════════════════════════════════════════════
    lmcache bench engine -- Interactive Setup
   ══════════════════════════════════════════════════

   Engine URL
     URL of the inference engine.
     [default: http://localhost:8000] >

   Workload
     The type of benchmark workload to run.
     Use up/down to navigate, Enter to select.

     * long-doc-qa           Repeated Q&A over long documents
       multi-round-chat       Multi-turn chat with stateful sessions
       long-doc-permutator    Permutations of context documents
       prefix-suffix-tuner    Two-pass tiered KV-cache demonstrator
       random-prefill         Prefill-only requests fired simultaneously

   LMCache Server
     Do you have a running LMCache server?
     It can auto-detect KV cache size information.
     [default: Y] (Y/n) >

   ...

   ──────────────────────────────────────────────────
    Configuration Summary
   ──────────────────────────────────────────────────
     Workload:             long-doc-qa
     Model:                Qwen/Qwen3-14B
     Tokens per GB:        6553
     ...
   ──────────────────────────────────────────────────

   What would you like to do?
     * Start benchmark
       Export configuration for later use and exit

When you choose "Export configuration", all auto-detected values (model name,
tokens per GB) are resolved and saved to a portable JSON file.


Config File
~~~~~~~~~~~

Config files store benchmark parameters but **not** environment-specific
values like engine URL or LMCache URL. This lets you reuse the same config
across different environments.

You can create a config file in three ways:

1. **Interactive mode** -- choose "Export configuration" at the summary step.
2. **``--export-config``** -- resolve and export from CLI without running.
3. **Manually** -- write JSON with keys matching CLI arg names (dashes
   replaced by underscores).

Example config file:

.. code-block:: json

   {
     "model": "Qwen/Qwen3-14B",
     "workload": "long-doc-qa",
     "tokens_per_gb_kvcache": 6553,
     "kv_cache_volume": 100.0,
     "ldqa_document_length": 10000,
     "ldqa_query_per_document": 2,
     "ldqa_shuffle_policy": "random",
     "ldqa_num_inflight_requests": 3
   }

Load it with ``--config`` (engine URL must be provided separately):

.. code-block:: bash

   lmcache bench engine --engine-url http://localhost:8000 \
       --config my_bench.json

CLI arguments override config file values, so you can use a base config and
tweak individual settings:

.. code-block:: bash

   # Use saved config but override KV cache volume
   lmcache bench engine --engine-url http://localhost:8000 \
       --config my_bench.json --kv-cache-volume 200


Output
~~~~~~

Terminal (real-time progress)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

During the benchmark, a live progress display shows in-flight requests,
average TTFT, decode speed, and throughput. Suppress it with ``-q``.

Terminal (final summary)
^^^^^^^^^^^^^^^^^^^^^^^^

After completion, a summary table is printed:

.. code-block:: text

   ======= Engine Benchmark Result (long-doc-qa) ========
   ---------------------- Configuration ------------------
   Engine URL:                       http://localhost:8000
   Model:                            Qwen/Qwen3-14B
   Workload:                         long-doc-qa
   ------------------------- Results ---------------------
   Successful requests:              20
   Failed requests:                  0
   Benchmark duration (s):           31.34
   Total input tokens:               200000
   Total output tokens:              2560
   Input throughput (tok/s):         6381.62
   Output throughput (tok/s):        81.69
   --------------- Time to First Token -------------------
   Mean TTFT (ms):                   313.41
   P50 TTFT (ms):                    272.83
   P90 TTFT (ms):                    587.21
   P99 TTFT (ms):                    837.32
   ------------------ Decoding Speed ---------------------
   Mean decode (tok/s):              48.23
   P99 decode (tok/s):               38.55
   ======================================================

CSV and JSON
^^^^^^^^^^^^

- ``bench_results.csv`` -- per-request metrics (TTFT, latency, decode speed,
  token counts). Written by default; skip with ``--no-csv``.
- ``bench_summary.json`` -- aggregate statistics with percentiles and config
  metadata. Opt-in with ``--json``.

Both files are written to ``--output-dir`` (default: current directory).


Exit Codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All requests succeeded.
   * - ``1``
     - One or more requests failed.


.. _lmcache-bench-server:

server
------

The ``lmcache bench server`` command is an end-to-end sanity test for the
LMCache Multi-Process (MP) cache server. It connects to a running server
over ZMQ and exercises the full KV-cache data path for a sequence of
synthetic requests, then optionally verifies per-chunk checksums through
the HTTP API.

.. code-block:: bash

   lmcache bench server [options]

Unlike :ref:`lmcache bench engine <lmcache-bench-engine>`, this command does
**not** require an inference engine. It only needs a running LMCache MP
server (ZMQ + HTTP). GPU mode additionally requires a CUDA-capable device.
It also requires the full ``lmcache`` install (not the lightweight
``lmcache-cli`` package).


What it does
~~~~~~~~~~~~

For each sequence in ``[--start, --end)``, the tool runs two passes:

1. **Cold pass** -- ``LOOKUP`` is expected to miss, so the generated KV
   tensors are ``STORE``\ d on the server.
2. **Warm pass** -- ``LOOKUP`` is expected to hit; the tool issues
   ``RETRIEVE`` and compares the retrieved KV chunks' checksums to the
   originals.

The full RPC path exercised is::

   REGISTER_KV_CACHE → GET_CHUNK_SIZE → LOOKUP
     → QUERY_PREFETCH_STATUS → RETRIEVE → STORE
     → END_SESSION

When ``--url`` points to the server's HTTP endpoint, per-chunk checksums
are additionally cross-checked against the server-side computation, so a
mismatch between producer and consumer surfaces as a loud
``CHECKSUM MISMATCH`` log line.


Quick start
~~~~~~~~~~~

Start the MP server in one terminal:

.. code-block:: bash

   lmcache server \
       --host localhost --port 15556 \
       --chunk-size 256 --l1-size-gb 5 \
       --eviction-policy LRU --max-workers 1

Then in another terminal:

.. code-block:: bash

   lmcache bench server \
       --rpc-url tcp://localhost:15556 \
       --url http://localhost:8080

By default the tool runs forever (``--end`` unset); stop it with
``Ctrl-C`` at any time. Pass ``--end N`` for a bounded run.


Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--rpc-url URL``
     - ``tcp://localhost:5555``
     - ZMQ endpoint of the MP cache server.
   * - ``--url URL``
     - ``http://localhost:8080``
     - HTTP base URL of the server's checksum API. Used to
       verify per-chunk checksums end-to-end.
   * - ``--mode {gpu,cpu}``
     - ``gpu``
     - Run mode. ``gpu`` allocates real CUDA tensors and uses CUDA IPC
       (lmcache-driven handle path). ``cpu`` allocates POSIX-SHM-backed tensors
       and uses the engine-driven (worker-side gather/scatter) path by default.
   * - ``--transfer-mode {auto,engine_driven,lmcache_driven}``
     - ``auto``
     - Transport routing for STORE/RETRIEVE. ``lmcache_driven`` forces the
       single-shot handle path (``REGISTER_KV_CACHE`` + ``STORE``/``RETRIEVE``),
       which supports both CUDA IPC and CPU SHM zero-copy transfers.
       ``engine_driven`` forces the worker-side gather/scatter path
       (``REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` + ``PREPARE``/``COMMIT``).
       ``auto`` maps gpu→lmcache_driven and cpu→engine_driven.
       The engine-driven data path is currently supported **only** for the
       historical ``--op pair --concurrency 1`` sanity test (its
       scatter/gather self-check is a discriminating oracle only there),
       and only on ``--mode cpu`` with ``--checksum on`` — that self-check
       is the sole oracle that can catch a dropped copy or an empty commit
       payload, so the data path may not run unverified (and
       ``--mode gpu --transfer-mode engine_driven`` is refused). A
       throughput op or ``--concurrency > 1`` on the data path is refused
       before any connection opens. Use the handle path (``--mode gpu`` or
       ``--transfer-mode lmcache_driven`` on cpu) for those. Engine-driven
       concurrency / throughput is a deferred follow-up.
       Even within that guard, engine-driven N=1 pair is a **compatibility
       sanity path**, not a general one: the validated configuration is CPU
       POSIX-SHM with a single homogeneous classical (5D) KV group. Other
       layouts (MLA / 4D or heterogeneous groups, or a no-SHM pickle
       fallback) may fail closed — the checksum catches any silent wrong
       result, but they are outside this path's tested scope. Use the
       handle path for those.
   * - ``--num-tokens N``
     - ``512``
     - Tokens per synthetic request.
   * - ``--num-blocks N``
     - ``1024``
     - Number of paged blocks allocated on the GPU.
   * - ``--block-size N``
     - ``16``
     - Tokens per paged block.
   * - ``--start N``
     - ``0``
     - First sequence number to run.
   * - ``--end N``
     - *(unset)*
     - Exclusive upper bound on sequence numbers. When omitted the
       loop runs forever.
   * - ``--interval SECS``
     - ``0.5``
     - Delay between successive sub-passes (``pair`` / ``retrieve-only``
       only; the throughput modes do not sleep between requests).
   * - ``--concurrency N``
     - ``1``
     - Number of concurrent worker threads, each with its own MP client
       and registered context. Each worker drives a disjoint sequence
       range and its own band-size KV cache (see
       `Concurrency and workload`_). ``1`` preserves the historical
       single-client request topology and default workload semantics.
   * - ``--op {pair,store-only,retrieve-only}``
     - ``pair``
     - Workload shape. ``pair`` is the historical cold-STORE + warm-RETRIEVE
       pass with checksum comparison. ``store-only`` runs a single cold
       STORE pass (write throughput). ``retrieve-only`` pre-warms the cache
       with a STORE pass, then measures a warm RETRIEVE pass (read
       throughput).
   * - ``--requests M``
     - *(unset)*
     - Requests **per worker**. When set, each worker issues exactly ``M``
       requests, so the run processes ``concurrency * M`` distinct
       sequences. Mutually exclusive with ``--end``.
   * - ``--prefetch-poll-interval SECS``
     - ``0.05``
     - Seconds between ``QUERY_PREFETCH_STATUS`` polls while waiting for a
       warm LOOKUP to become ready. Must be finite and ``>= 0``. The 50 ms
       default dominates the per-request latency floor at high concurrency;
       lower it to observe the server's true prefetch latency. The
       inter-poll sleep budget is ``50 * interval``; the total LOOKUP wall
       time also includes the per-poll RPC latency and, in the worst case,
       the RPC timeout.
   * - ``--checksum {auto,on,off}``
     - ``auto``
     - Checksum verification mode. ``auto`` enables checksums in ``pair``
       mode and disables them in the throughput modes so a run is not
       skewed by the per-request checksum round-trip. ``on`` / ``off``
       force the choice, but ``on`` is only valid with ``--op pair``:
       the throughput modes have no checksum contract yet (a follow-up
       will add an independent read-back verification pass), so
       ``--op store-only``/``retrieve-only`` with ``--checksum on`` is
       rejected.
       In handle mode (``lmcache_driven``) the warm pass zero-fills the
       client's shared KV pages before the RETRIEVE, so the post-retrieve
       server-side digest can only match the cold digest if the server
       actually wrote the bytes back — a silent no-op RETRIEVE fails the
       comparison instead of trivially passing it.
   * - ``--server-max-gpu-workers M``
     - *(unset)*
     - The AFFINITY transfer-pool size the target server was started with.
       The server sizes that pool from its ``max_gpu_workers`` setting
       (``add_affinity_thread_pool(max_workers=max_gpu_workers)``) and does
       not expose it over RPC, so supply it explicitly; it is recorded in
       the results config section. **Required when** ``--concurrency > 1``,
       and the run is refused when concurrency exceeds it (see
       `Concurrency and workload`_).
   * - ``--server-commit REV``
     - *(empty)*
     - Free-form identifier of the server build under test (e.g. its git
       commit), recorded verbatim in the results config section.
   * - ``--server-image IMAGE``
     - *(empty)*
     - Free-form identifier of the server container image / environment,
       recorded verbatim in the results config section.
   * - ``--kvcache-shape-spec SPEC``
     - ``(2,1024,16,8,128):float16:32``
     - KV cache shape spec (see below).
   * - ``--format FORMAT``
     - ``terminal``
     - Stdout output format for the final metrics summary. Available:
       ``terminal``, ``json``.
   * - ``--output PATH``
     - *(unset)*
     - Save the final metrics summary to a file at PATH (format chosen
       by ``--format``).
   * - ``-q`` / ``--quiet``
     - *(unset)*
     - Suppress all progress messages during the run. Only the final
       structured metrics summary is emitted (unless also redirected
       via ``--output``).


Concurrency and workload
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default the bench drives one request at a time. ``--concurrency N``
runs ``N`` worker threads. Each worker owns its **own** MP client and its
**own** registered KV-cache context, so the workers drive ``N``
independent transfer lanes:

* **Own client.** Each worker's ZMQ ``DEALER`` socket gets a distinct
  identity, which the server uses as the affinity key for its
  transfer thread pool. A single shared client would pin every worker's
  STORE / RETRIEVE onto one affinity worker and serialize them; distinct
  identities allow requests to be distributed across the server's
  affinity workers. The server maps an affinity key to a slot on the
  key's **first appearance**, round-robin over the pool; the mapping is
  never removed and slots are never freed, so there is no "free slot"
  to reclaim. Distinct slots for all ``N`` bench workers are therefore
  guaranteed under exactly this condition: on a **dedicated server** (no
  other affinity client has contacted it since it started), the bench's
  ``N`` fresh identities appear first and ``N <= max_gpu_workers``. The
  server cannot be queried for ``max_gpu_workers``, so ``--concurrency > 1``
  requires ``--server-max-gpu-workers``; the run is refused when ``N``
  exceeds it, and the value is recorded in the results config section.
  (The bench clients still share one background polling thread — see the
  throughput caveat under `Output`_.)
* **Own context (band-size KV cache).** Worker ``w`` registers under
  ``instance_id = instance_base + w`` (``instance_base`` is a fresh
  per-run 62-bit nonce, which practically avoids accidentally reusing a
  prior run's still-registered context) with a KV cache of one **band**
  (``--num-blocks / N`` blocks). Because the pool is split ``N`` ways
  rather than replicated, the total registered KV tensor capacity never
  exceeds the requested pool and stays approximately constant as
  concurrency grows — the floor division may leave up to ``N - 1``
  blocks unused (e.g. ``1000 // 3 * 3 = 999``) — while per-client and
  per-context metadata overhead still scales with ``N``. Each worker
  addresses only its own tensors, so two workers can never race on the
  same memory. Each worker also fills its band from its own
  deterministic seed (``42 + worker_id``), so no two bands hold
  identical bytes and a retrieve that lands in the wrong worker's band
  cannot silently pass a checksum comparison.
* **Sequence space.** Worker ``w`` processes sequence numbers
  ``start + w``, ``start + w + N``, ... (stride ``N``). Since the cache
  key is derived from the sequence number, workers never collide on a
  server-global key.

The run fails fast if the pool cannot be divided — each worker's band must
be at least one request wide
(``--num-blocks / --concurrency >= tokens-per-request / --block-size``).
Reduce ``--concurrency``, raise ``--num-blocks``, or lower ``--num-tokens``
if you hit this.

``--concurrency 1`` preserves the historical single-client request
topology and default workload semantics: one client, one registered
context, and the full pool as the single band (the block-offset formula
matches the historical one — unit-tested against it). The registered
``instance_id`` is still the per-run nonce (``instance_base + 0``), not a
fixed ``0``, so back-to-back single-worker runs do not collide either.

Concurrent runs must be bounded: pass ``--requests M`` (M requests per
worker) or ``--end N``. Unbounded (forever) runs are only allowed at
``--concurrency 1``.

Concurrent phases use a two-phase start: every worker thread is created
first and blocks on a start gate; the wall clock (and the server
profiler) starts only once all workers have signalled ready, and only
then is the gate released. Thread-startup skew is therefore excluded
from the measured wall time, and the run has true ``N``-way overlap from
the first request onward.

If **any** worker raises, the run fails: the first exception stops the
remaining workers, teardown still deregisters every context, and the tool
exits ``1`` with the summary marked invalid (see `Output`_).

Every request must also satisfy the **contract of its workload mode**;
any violation, operation failure, or RPC timeout aborts the run the same
way (summary ``Valid: no``, exit ``1``). The failure counters are still
printed for diagnosis, but an invalid run never reports throughput or
latency sections:

* In every mode, a LOOKUP timeout or a prefetch-status poll failure is a
  request failure — a poll timeout is **never** treated as a cache miss.
* ``store-only`` (and the ``retrieve-only`` pre-warm): every request
  must be a full miss and STORE successfully; the pass never issues a
  RETRIEVE. An unexpected hit (for example, keys left over from a
  previous run) is a contract violation — use a fresh ``--start`` range
  or restart the server.
* The measured ``retrieve-only`` pass: every request must be a full hit
  and RETRIEVE successfully; the pass never issues a STORE.
* ``pair``: the hit portion must RETRIEVE and the miss portion must
  STORE successfully, and any checksum mismatch marks the run invalid.
* A failed or timed-out ``UNREGISTER`` during teardown also invalidates
  the run: a context the server still holds skews every subsequent run
  against this server.
* Some failures also **taint the server**: a prefetch-status poll timeout
  (the server keeps the prefetch job — ``END_SESSION`` does not cancel it),
  or a timed-out cleanup RPC (``END_SESSION`` / ``FREE_LOOKUP_LOCKS``,
  which may leave a session or read locks held). A run nonce avoids id
  *collisions* across runs but cannot reclaim leaked *resources*, so such a
  run reports ``Server reuse safe: no`` and the dedicated server **must be
  restarted** before the next run.

The ``--op`` flag selects what each worker measures:

* ``pair`` (default) — the historical cold STORE then warm RETRIEVE per
  sequence, with a checksum comparison. Latency-oriented; ``--interval``
  applies, so its ops/s is interval-bound rather than a true throughput
  number.
* ``store-only`` — a single cold STORE pass over unique sequences. No
  inter-request sleep; use for write throughput.
* ``retrieve-only`` — read throughput. Runs in two phases separated by a
  global barrier: **Phase A** pre-warms the cache with a STORE pass across
  all workers and joins; **Phase B** then measures a warm RETRIEVE pass.
  The pre-warm is excluded from the reported wall time and from the server
  profiler, so only the measured retrieves count. Phase B starts only
  when the pre-warm provably completed: every worker finished all of its
  expected requests and every one of them was a full-miss, successful
  STORE. Any pre-warm failure, timeout, or short completion aborts the
  run as invalid before a single measured RETRIEVE is issued.

For a concurrency sweep, run the command once per level from a shell loop
and collect the ``--format json`` output. Pass the server's actual
``max_gpu_workers`` via ``--server-max-gpu-workers`` — a level whose ``N``
exceeds it is refused. Give each level a
**non-overlapping** ``--start`` so the sequence numbers (and therefore the
cache keys) never collide between levels — otherwise a later level would
warm-hit data an earlier level already stored and its "cold" STORE pass
would measure retrieves instead:

.. code-block:: bash

   # c=1 uses seqs [0, 200); c=2 uses [10000, 10400); c=4 uses [20000, ...)
   # Throughput / concurrency runs on cpu need the handle path, so pass
   # --transfer-mode lmcache_driven (the default cpu auto maps to the
   # engine-driven data path, which is refused for store-only / N>1).
   start=0
   for c in 1 2 4; do
     lmcache bench server --rpc-url tcp://localhost:5555 \
         --mode cpu --transfer-mode lmcache_driven \
         --op store-only --concurrency "$c" --requests 200 \
         --start "$start" --server-max-gpu-workers 8 \
         --format json --output "sweep-c$c.json"
     start=$((start + 10000))
   done

.. note::

   Restarting the CLI between sweep levels does **not** clear server-side
   or L2 cache state — the server keeps everything the previous level
   stored. The non-overlapping ``--start`` above is what keeps the levels
   independent. A future revision may add ``--cache-salt`` / ``--run-id``
   flags to namespace cache keys per run automatically; until then, manage
   the key space yourself with ``--start``, or restart the server between
   sweeps if you need a truly cold cache.


CPU mode (no GPU)
~~~~~~~~~~~~~~~~~

``--mode cpu`` runs the same end-to-end path without a GPU. The server
runs on a CPU-only host (``StubCPUDevice``); the bench tool allocates
POSIX-SHM-backed KV tensors and exercises the full RPC path.

By default ``--mode cpu`` uses the engine-driven gather/scatter path
(``auto`` → ``cpu→engine_driven``). To use the zero-copy SHM
handle path instead, pass ``--transfer-mode lmcache_driven``:

.. code-block:: bash

   # Terminal 1 -- start the LMCache server (no GPU required)
   lmcache server \
       --host localhost --port 5555 \
       --l1-size-gb 2 --eviction-policy LRU

   # Terminal 2 -- run bench in CPU + lmcache_driven mode
   lmcache bench server \
       --rpc-url tcp://localhost:5555 \
       --url http://localhost:8080 \
       --mode cpu --transfer-mode lmcache_driven \
       --start 0 --end 2


KV cache shape spec
~~~~~~~~~~~~~~~~~~~

The ``--kvcache-shape-spec`` flag describes how KV tensors are laid out on
the GPU. A spec is one or more groups separated by ``;``:

.. code-block:: text

   (kv_size,NB,BS,NH,HS):dtype:layers[;(...):dtype:layers...]

Fields:

* ``kv_size`` -- 2 for classical attention (separate K/V), 1 for MLA.
* ``NB`` -- number of paged blocks.
* ``BS`` -- block size (tokens per block).
* ``NH`` -- number of attention heads per layer.
* ``HS`` -- head size (in elements).
* ``dtype`` -- element dtype (e.g. ``float16``, ``bfloat16``, ``float32``,
  ``uint8``). The full set matches the keys of ``DTYPE_MAP`` in
  ``lmcache/v1/kv_layer_groups.py``.
* ``layers`` -- number of layers in this group.

Multi-group specs let you model heterogeneous layers (for example, MLA
layers + classical attention layers in the same model):

.. code-block:: bash

   lmcache bench server \
       --rpc-url tcp://localhost:15556 \
       --kvcache-shape-spec "(1,1024,16,1,128):float16:4;(2,1024,16,8,128):float16:28"

All groups must share the same ``NB`` and ``BS`` (this is a physical
constraint of paged KV). Layer counts across groups sum to the total
layer count registered with the server.

See ``parse_kvcache_shape_spec`` in ``lmcache/v1/kv_layer_groups.py``
for the authoritative parsing rules and validation errors.


Profiling the server
~~~~~~~~~~~~~~~~~~~~~

``lmcache bench server`` is a ZMQ client: the store path it exercises
(hashing, allocation, gather, D2H) runs inside the **server** process, not
this benchmark. ``--flamegraph on`` therefore attaches the profiler to a
server pid you supply, records for the duration of the load, and renders a
flame graph of the server, not of the client.

.. code-block:: bash

   lmcache bench server \
       --rpc-url tcp://localhost:5555 \
       --start 0 --end 200 --interval 0.02 \
       --flamegraph on --flamegraph-mode gil \
       --profile-server-pid "$(pgrep -f 'lmcache server')"

``--flamegraph-mode`` takes the same six values documented under
:ref:`lmcache tool flamegraph <lmcache-flamegraph-modes>` (or several
comma-separated to drive the load once per mode, one SVG each). Because the
target is a separate, already-running server (not a process this benchmark
spawns), it profiles by *attaching*, so the same attach-mode caveats
documented for
:doc:`lmcache tool flamegraph </cli/tool>` apply here: what each mode
shows, the ``PYTHONPERFSUPPORT=1`` requirement for naming Python frames in
the perf/bcc modes, the container privileges each mode needs, and the fact
that recording a live process is never free.

.. note::

   The one thing unique to ``bench server``: it records *while* it drives
   load, so the recording overhead lands on the very throughput/latency
   this benchmark reports. Keep the profiled run short and read those
   numbers as indicative, not a clean baseline.


Output
~~~~~~

After the run completes (or is interrupted with ``Ctrl-C``), a structured
metrics summary is printed. The summary includes:

* **Configuration** -- RPC URL, mode, transfer mode, op, concurrency,
  tokens per request, interval, the ``instance_id_base`` run nonce (the
  base of every worker's registered context id), and the
  operator-supplied server metadata: ``server_max_gpu_workers``
  (``unknown`` when not given) plus ``server_commit`` / ``server_image``
  when provided, so a result file is attributable to a specific server
  configuration.
* **Results** -- whether the run is valid, completed-vs-expected worker
  counts, total requests, and STORE / RETRIEVE failure accounting
  (attempted / OK / failed / timeouts for whichever operations the
  workload issued). Pair mode additionally reports checksum OK / FAIL
  counts and pass rate. If a worker aborted, an ``Error`` row is shown and
  the throughput / latency sections are omitted (the run is invalid).
* **Throughput** -- measured-phase wall time, requests/s, the number of
  MB **successfully** stored / retrieved, and store / retrieve MB/s. Only
  successful transfers count toward the bytes; failed or timed-out ones
  are excluded. ops/s and MB/s divide by the measured-phase wall time only
  (the one-time register / unregister, and for ``retrieve-only`` the
  pre-warm, are excluded). These are **client-observed end-to-end**
  numbers, not a pure server-side measurement: all ``N`` bench workers
  share a single background ZMQ polling thread that serializes message
  send / receive and encode / decode, which can become the client-side
  ceiling at high concurrency. If MB/s plateaus as ``N`` grows, record
  the bench process's polling-thread CPU usage before concluding that
  the server saturated.
* **Latency sections** -- per-operation latency statistics (count, mean,
  min, max, p50, p95, p99) for cold lookup, cold store, warm lookup, and
  warm retrieve. Under concurrency the percentiles are pooled across all
  workers.

Use ``--format json`` to get machine-readable output, or ``--output FILE``
to save the summary to a file.

.. code-block:: text

   ================ Server Bench Result =================
   ---------------------- Configuration -----------------
   RPC URL:                          tcp://localhost:15556
   Mode:                             gpu
   Transfer mode:                    auto
   Op:                               pair
   Concurrency:                      1
   Tokens / request:                 512
   Interval (s):                     0.5
   Instance ID base:                 1234567890
   Server max GPU workers:           unknown
   ------------------------- Results --------------------
   Valid:                            yes
   Completed workers:                1
   Expected workers:                 1
   Total requests:                   3
   Store attempted:                  3
   Store OK:                         3
   Store failed:                     0
   Store timeouts:                   0
   Retrieve attempted:               3
   Retrieve OK:                      3
   Retrieve failed:                  0
   Retrieve timeouts:                0
   Checksum OK:                      3
   Checksum FAIL:                    0
   Pass rate (%):                    100.0
   ----------------------- Throughput -------------------
   Wall time (s):                    3.041
   Successful store (MB):            65.6
   Successful retrieve (MB):         65.6
   Requests/s:                       0.99
   Store MB/s:                       21.6
   Retrieve MB/s:                    21.6
   -------------------- Cold Lookup (ms) ---------------
   count:                            3
   mean:                             1.647
   min:                              1.312
   max:                              1.823
   p50:                              1.647
   p95:                              1.823
   p99:                              1.823
   --------------------- Cold Store (ms) ---------------
   count:                            3
   mean:                             1.740
   min:                              1.521
   max:                              1.982
   p50:                              1.740
   p95:                              1.982
   p99:                              1.982
   -------------------- Warm Lookup (ms) ---------------
   count:                            3
   mean:                             1.310
   min:                              1.102
   max:                              1.512
   p50:                              1.310
   p95:                              1.512
   p99:                              1.512
   ------------------- Warm Retrieve (ms) --------------
   count:                            3
   mean:                             1.480
   min:                              1.321
   max:                              1.612
   p50:                              1.480
   p95:                              1.612
   p99:                              1.612
   =====================================================


Example output (progress)
~~~~~~~~~~~~~~~~~~~~~~~~~

During the run, ``pair`` mode prints per-request progress to stdout
(suppressed by ``-q`` / ``--quiet``). The throughput modes
(``store-only`` / ``retrieve-only``) stay silent by default so the chatty
per-request lines do not distort a high-rate run; only the final summary
is printed. Warnings and checksum mismatches always go to stderr.

.. code-block:: text

   Connecting to LMCache MP Server at tcp://localhost:15556 (mode=gpu) ...
   Server chunk_size = 256
   Resolved KV shape spec: (2,1024,16,8,128):float16:32
   === [w0] Request seq=0 ===
     [w0 seq 0/cold] LOOKUP: 0/2 chunks hit (1.8 ms)
     [w0 seq 0/cold] STORE: stored (512 tokens, 1.7 ms)
     [w0 seq 0/warm] LOOKUP: 2/2 chunks hit (1.3 ms)
     [w0 seq 0/warm] RETRIEVE: retrieved (512 tokens, 1.5 ms)
     [w0 seq 0] CHECKSUM MATCH OK
   === [w0] Request seq=1 ===
   ...

The ``w0`` prefix is the worker index; under ``--concurrency N`` the lines
from all workers interleave. Any ``CHECKSUM MISMATCH``, ``ERROR``, or
Python traceback (on stderr) indicates a real problem worth investigating.


Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All workers completed (or the run was interrupted cleanly with
       Ctrl-C) with no worker errors.
   * - ``1``
     - Fatal error or an invalid run: CUDA unavailable in ``--mode gpu``,
       server unreachable, a failed registration, any worker raising,
       any request-level LOOKUP / prefetch-poll / STORE / RETRIEVE
       failure or timeout, a workload-contract violation, a checksum
       mismatch, or a failed teardown ``UNREGISTER``.

.. _lmcache-bench-l2:

l2
--

The ``lmcache bench l2`` command benchmarks an L2 cache adapter
(e.g. the local-filesystem adapter) end-to-end through the same
``parse_args_to_l2_adapters_config`` + ``create_l2_adapter`` pipeline that
LMCache uses in production. Any registered adapter type can be tested
without code changes: you describe the adapter with a single JSON spec
and pick the operations to exercise.

.. code-block:: bash

   lmcache bench l2 [options]

Unlike :ref:`lmcache bench engine <lmcache-bench-engine>`, this command
does **not** require an inference engine or an LMCache MP server. It
only needs the adapter's own backing storage to be reachable (for the
``fs`` adapter, that simply means a writable directory).


What it does
~~~~~~~~~~~~

For each measured operation the tool drives the adapter directly via
its public submit/wait API:

* ``Store``  -- ``submit_store_task`` writes ``num_keys`` MemoryObjs per
  submit and waits for the store eventfd.
* ``Lookup`` -- ``submit_lookup_and_lock_task`` checks key existence
  (no payload transfer) and waits for the lookup eventfd.
* ``Load``   -- ``submit_load_task`` reads ``num_keys`` MemoryObjs per
  submit and waits for the load eventfd.

Each measured **round** issues ``--in-flight`` submits sequentially from
a single producer thread and then waits for all of them to complete; the
round duration is the wall-clock time from the first submit until the
last completion. Warmup rounds run before measurement and their results
are discarded from the final summary.

All three operations share the same key idx universe, so running
``--only store`` followed by ``--only load`` (or ``--only lookup``) with
identical other flags hits exactly the same keys. This makes the
benchmark useful as a quick regression test for adapters that should
support a clean store -> load round-trip.

.. note::

   When ``--only`` is not given, the three operations are run **in a
   single process in the order** ``store -> lookup -> load``. For
   adapters whose backing storage sits behind an OS-level cache --
   most notably the local-filesystem (``fs``) adapter, which is
   subject to the Linux **page cache** -- this means ``lookup`` and
   ``load`` will almost always observe the data that ``store`` just
   wrote still hot in RAM, and the reported numbers reflect
   page-cache throughput rather than the underlying device.

   To benchmark each operation against a cold cache, run them
   separately with ``--only`` and drop the OS caches in between, for
   example::

      lmcache bench l2 --l2-adapter '...' --only store
      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
      lmcache bench l2 --l2-adapter '...' --only lookup
      sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
      lmcache bench l2 --l2-adapter '...' --only load

   For adapters that bypass the page cache (e.g. ``fs`` with
   ``"use_odirect": true``) or that talk to a remote service without
   a local cache, the default combined run is usually fine.

   O_DIRECT adapters may also require the benchmark L1 buffer to
   satisfy the adapter's block alignment. Use ``--l1-align-bytes`` to
   set that alignment, commonly ``4096`` for local block devices. The
   payload size (``--data-size-kb * 1024``) must be a multiple of the
   selected alignment.


Quick start
~~~~~~~~~~~

Benchmark the local filesystem adapter with default parameters:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}'

This runs all three operations (store, lookup, load) with one warmup
round and one measurement round.

Stress the adapter with more in-flight submits and larger payloads:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/data/lmcache-bench","relative_tmp_dir":"tmp"}' \
       --num-keys 32 --in-flight 4 \
       --data-size-kb 512 \
       --rounds 5 --warmup-rounds 1

Benchmark an O_DIRECT adapter with aligned L1 buffers:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"raw_block","device_path":"/dev/nvme0n1","slot_bytes":4194304,"use_odirect":true,"block_align":4096}' \
       --data-size-kb 1024 \
       --l1-align-bytes 4096

Run only one operation (useful to isolate store vs. load throughput):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --only store

Lookup with a controlled hit rate (the benchmark splits the lookup keys
between a potentially-existing range and a guaranteed-non-existent
range):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --only lookup --lookup-max-hit-rate 0.5

Enable a store -> load round-trip data integrity check on the last
measured round:

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/tmp/lmcache-bench"}' \
       --no-skip-verify

If you prefer to keep the JSON spec out of the command line, set the
``L2_ADAPTER_JSON`` environment variable instead of passing
``--l2-adapter``:

.. code-block:: bash

   export L2_ADAPTER_JSON='{"type":"fs","base_path":"/tmp/lmcache-bench"}'
   lmcache bench l2 --num-keys 32 --in-flight 2


Options
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Flag
     - Default
     - Description
   * - ``--l2-adapter JSON``
     - *(unset)*
     - L2 adapter spec as JSON with a ``"type"`` field plus
       adapter-specific configs, e.g.
       ``'{"type":"fs","base_path":"/tmp/bench"}'``. May be passed
       multiple times; only the first spec is benchmarked. If not
       provided, falls back to the ``L2_ADAPTER_JSON`` environment
       variable. Either the flag or the env var is **required**.
   * - ``--num-keys N``
     - ``32``
     - Number of keys per submit.
   * - ``--in-flight N``
     - ``1``
     - In-flight submits per round. Each round issues this many
       submits sequentially from a single producer thread, then waits
       for all of them.
   * - ``--data-size-kb N``
     - ``256``
     - Data size per key, in KiB.
   * - ``--l1-align-bytes N``
     - ``1``
     - Alignment in bytes for benchmark L1 buffers. Use a value
       at least as large as the adapter's block alignment when
       benchmarking O_DIRECT backends, for example ``4096`` for local
       block devices. ``--data-size-kb * 1024`` must be a multiple of
       this value.
   * - ``--rounds N``
     - ``1``
     - Measurement rounds per operation.
   * - ``--warmup-rounds N``
     - ``1``
     - Warmup rounds run before measurement; their results are
       discarded.
   * - ``--lookup-max-hit-rate F``
     - ``0.0``
     - Upper bound on the lookup hit rate, in ``[0, 1]``. The benchmark
       requests ``floor(N * rate)`` keys from the
       potentially-existing range and ``N - hit`` keys from a
       guaranteed-non-existent range, where ``N`` is the total number
       of lookup keys. The actual hit rate may be lower if those keys
       were never stored in this run.
   * - ``--skip-verify`` / ``--no-skip-verify``
     - ``--skip-verify``
     - Skip the store -> load round-trip data integrity check (the
       default). Pass ``--no-skip-verify`` to enable verification on
       the last measured round; this requires both ``store`` and
       ``load`` to be exercised.
   * - ``--only {lookup,store,load}``
     - *(unset)*
     - Run only the specified operation. When omitted, all three
       operations are run in the order ``store -> lookup -> load``.
   * - ``--flamegraph {on,off}``
     - ``off``
     - Capture a flame graph of the measured phases (``on``) or run the
       benchmark normally (``off``). When ``on``, the benchmark profiles
       itself and renders an SVG. Default ``off`` leaves benchmark
       behavior unchanged. See
       :ref:`Profiling / flame charts <lmcache-bench-l2-profiling>`.
   * - ``--flamegraph-mode {on-cpu,off-cpu,wakeup,offwake,wall,gil}``
     - ``on-cpu``
     - Flame-graph mode for ``--flamegraph on``. ``on-cpu`` shows
       where CPU time goes; ``off-cpu`` shows time blocked on I/O /
       locks (best for I/O-bound adapters); ``offwake`` adds the waker
       stack to each blocked stack; ``wakeup`` shows the stacks doing
       the waking. ``wall`` and ``gil`` (``py-spy``) split the chart
       per thread: wall-clock time, and time holding the interpreter
       lock.
   * - ``--flamegraph-output PATH``
     - *(auto)*
     - SVG output path. Default:
       ``/tmp/lmcache_bench_flames/<adapter>.<mode>.svg``.
   * - ``--flamegraph-scripts-dir DIR``
     - *(~/FlameGraph)*
     - Directory with the FlameGraph scripts (``flamegraph.pl``,
       ``stackcollapse-perf.pl``).


Adapter JSON spec
~~~~~~~~~~~~~~~~~

The ``--l2-adapter`` JSON is parsed by
``lmcache.v1.distributed.l2_adapters.config.parse_args_to_l2_adapters_config``,
the same entry point LMCache uses everywhere else. The minimum required
field is ``type``; all remaining fields are forwarded to the adapter
implementation as keyword arguments.

Example for the local-filesystem adapter:

.. code-block:: json

   {
     "type": "fs",
     "base_path": "/data/lmcache-bench",
     "relative_tmp_dir": "tmp",
     "read_ahead_size": null,
     "use_odirect": false
   }

See the source under ``lmcache/v1/distributed/l2_adapters/`` for the
full list of adapter types and their accepted fields.


Example output
~~~~~~~~~~~~~~

Per-round progress (suppressed by ``-q``):

.. code-block:: text

   ============================================================
   L2 Adapter Benchmark
   ============================================================
     Adapter config         : FSL2AdapterConfig
     L2 adapter JSON        : {"type":"fs","base_path":"/data/lmcache-bench","relative_tmp_dir":"tmp"}
     Keys / submit          : 32
     In-flight / round      : 3
     Keys / round           : 96
     Data size / key        : 256 KB
     Data / round           : 24.00 MB
     Rounds                 : 1 (+ 1 warmup)
     Lookup max hit rate    : 0.00%
   ============================================================

   [Init] Creating adapter...
   [Init] Adapter created successfully (FSL2Adapter).

   [Store] Running 1 warmup + 1 measurement rounds...
     [Store] Round 1: 47.83 ms, success_keys=96/96
     [Store] Round 2: 46.19 ms, success_keys=96/96

   [Lookup] Running 1 warmup + 1 measurement rounds...
     [Lookup] Round 1:  5.36 ms, found=96/96
     [Lookup] Round 2:  5.03 ms, found=96/96

   [Load] Running 1 warmup + 1 measurement rounds...
     [Load] Round 1: 18.15 ms, loaded=96/96
     [Load] Round 2: 17.63 ms, loaded=96/96

Final summary (one section per exercised operation):

.. code-block:: text

   ====== L2 Adapter Benchmark Result (FSL2Adapter) =======
   ----------------------- Configuration -------------------
   Adapter:                          FSL2Adapter
   Keys / submit:                    32
   In-flight / round:                3
   Data size / key (KB):             256
   Measurement rounds:               1
   Warmup rounds:                    1
   Lookup max hit rate:              0.0
   --------------------------- Store -----------------------
   Operation:                        Store
   Rounds:                           1
   Keys / round:                     96
   Total keys:                       96
   Total success:                    96
   Duration avg (ms):                46.19
   ...
   Throughput avg (MB/s):            519.62
   Avg ops/s:                        2078.50
   Avg latency / key (ms):           0.481
   --------------------------- Lookup ----------------------
   ...
   ---------------------------- Load -----------------------
   ...
   =========================================================

Each operation section reports per-round duration statistics
(avg / min / max / p50 / p99 / std), aggregate throughput
(``avg_throughput_mbps`` -- 0 for ``Lookup`` since it has no payload),
average key-rate (``avg_ops_per_sec``), and a per-key latency.

For ``Lookup``, three additional fields are reported when
``--lookup-max-hit-rate`` is non-zero or some keys were found:

* ``Expected max hit rate`` -- the configured upper bound.
* ``Expected hit keys`` -- ``floor(total_keys * rate)``, scaled for
  the measured rounds only.
* ``Actual hit rate`` -- the measured hit rate over the kept rounds.


Round-trip verification
~~~~~~~~~~~~~~~~~~~~~~~~

When ``--no-skip-verify`` is passed and both ``store`` and ``load`` were
run, the benchmark compares the load buffers from the last measured
round against the byte pattern that ``store`` wrote (see
``make_memory_objects`` in
``lmcache/cli/commands/bench/l2_adapter_bench/data.py``):

.. code-block:: text

   [Verify] Checking store -> load data integrity for last measured round...
   [Verify] OK

Verification is **off** by default because the stricter byte pattern
requires both the store and load object batches to stay resident so the
loaded data can be compared against the original store pattern.


.. _lmcache-bench-l2-profiling:

Profiling / flame charts
~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``--flamegraph on`` is passed, the benchmark profiles **its own
process** (the L2 adapter driven by this microbenchmark's synthetic load)
and renders a flame graph of the measured phases (to profile a separate
server or a real process instead, use
:doc:`lmcache tool flamegraph </cli/tool>`):

.. code-block:: bash

   lmcache bench l2 \
       --l2-adapter '{"type":"fs","base_path":"/data/lmcache-bench"}' \
       --rounds 300 --flamegraph on --flamegraph-mode on-cpu
   #   [Profile] on-cpu recording started (pid=12345) -> .../FSL2Adapter.oncpu.svg
   #   [Profile] wrote /tmp/lmcache_bench_flames/FSL2Adapter.oncpu.svg

The ``--flamegraph-mode`` values, cost of recording, and tool / sysctl
requirements are documented under
:ref:`lmcache tool flamegraph <lmcache-flamegraph-modes>` (or pass several
comma-separated to profile one benchmark run per mode, one SVG each). What is
specific to ``bench l2``:

* It **self-profiles**, so on CPython 3.12+ it activates the perf trampolines
  itself and adapter functions resolve as ``py::<qualname>`` in the
  ``on-cpu`` / ``off-cpu`` charts with no ``PYTHONPERFSUPPORT`` needed (an
  attached server cannot). Trampolines cost a few percent, so treat a
  profiled run's timings as indicative.
* The recorder runs as a child of the benchmark, so ``wall`` / ``gil`` need
  ``kernel.yama.ptrace_scope`` at ``0`` (not the attach-mode permissions).
* Recording covers only the measured work, so use a large ``--rounds``; too
  short a run captures no samples.

The SVG is written to ``--flamegraph-output`` (default
``/tmp/lmcache_bench_flames/<adapter>.<mode>.svg``).


Exit codes
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Code
     - Meaning
   * - ``0``
     - All requested operations completed and (when enabled) the
       round-trip verification passed.
   * - ``1``
     - Adapter creation failed, round-trip verification failed, or
       an operation hit a fatal error (e.g. all rounds timed out).
   * - ``2``
     - Invalid invocation: the ``--l2-adapter`` JSON / ``L2_ADAPTER_JSON``
       env var was missing or could not be parsed, an option value was
       invalid, or ``--flamegraph on`` was requested but the profiling
       toolchain is unavailable.
