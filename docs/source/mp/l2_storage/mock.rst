Mock
====

Simulates L2 storage with configurable size, bandwidth, and per-operation
service latency.  Useful for testing the L2 pipeline without real storage
hardware, and for benchmarking how backend service time affects the
LMCache-side pipeline.

**Fields:**

- ``max_size_gb``: Maximum size in GB (> 0).
- ``mock_bandwidth_gb``: Simulated bandwidth in GB/sec (> 0).
- ``mock_store_latency_ms``: Fixed service latency added to every store
  task, in milliseconds (>= 0, default 0).
- ``mock_lookup_latency_ms``: Fixed service latency added to every
  lookup_and_lock task, in milliseconds (>= 0, default 0).
- ``mock_load_latency_ms``: Fixed service latency added to every load
  task, in milliseconds (>= 0, default 0).

The fixed latencies model the size-independent cost of a real backend
(RPC round trip, metadata service, object-store HEAD), while
``mock_bandwidth_gb`` models the size-proportional transfer cost.
Together they define the *minimum modeled service time* of a task:
store and load complete no earlier than
``submit + latency + bytes / bandwidth``, with adapter-side work already
performed counting toward that target; lookup simulates the fixed
latency before processing the request.  The per-task latency applies
once per submitted batch, including no-op tasks.

.. code-block:: bash

    --l2-adapter '{"type": "mock", "max_size_gb": 256, "mock_bandwidth_gb": 10}'

Simulating a backend with 2 ms fixed service latency per operation:

.. code-block:: bash

    --l2-adapter '{"type": "mock", "max_size_gb": 256, "mock_bandwidth_gb": 10,
                   "mock_store_latency_ms": 2, "mock_lookup_latency_ms": 2,
                   "mock_load_latency_ms": 2}'
