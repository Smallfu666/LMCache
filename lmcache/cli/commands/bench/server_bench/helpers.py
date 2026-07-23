# SPDX-License-Identifier: Apache-2.0
"""Internal helpers for ``lmcache bench server``.

This module owns the heavy runtime imports (``torch`` / ``zmq`` /
``lmcache.v1.*``) and all pure / low-level helper functions used by
the ``server`` bench target. The CLI registration and execute
orchestration live in :mod:`lmcache.cli.commands.bench.server_bench.command`.

Splitting the module this way keeps the public command surface in line
with the ``engine_bench`` and ``l2_adapter_bench`` siblings, while
still quarantining the heavy imports behind a single guarded block so
the slim ``lmcache-cli`` install can load the bench parser without
torch / zmq.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any
import ctypes
import hashlib
import json
import mmap
import sys
import time
import urllib.error
import urllib.request

if TYPE_CHECKING:
    # Standard
    from collections.abc import Callable

# First Party
from lmcache import torch_dev, torch_device_type

# ``lmcache bench server`` allocates real CUDA tensors and talks to
# the MP server via ZMQ, both of which are absent from the thin
# ``lmcache-cli`` distribution (no torch, no zmq, no lmcache.v1.*).
# Importing them unconditionally would kill the *entire* ``lmcache``
# CLI at registry load time with an opaque ImportError. Wrap the
# heavy imports and remember the error so ``add_arguments`` /
# ``execute`` can bail out with an actionable install hint.
_IMPORT_ERROR: ImportError | None = None
try:
    # Third Party
    import torch
    import zmq  # noqa: F401  # availability probe; used by command.py

    # First Party
    from lmcache.utils import (
        EngineType,
        check_interprocess_event_support,
    )
    from lmcache.v1.kv_layer_groups import (
        DTYPE_MAP,
        KVLayerGroupInfo,
    )
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheServerKey,
        KVCache,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.futures import MessagingFuture
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap
    from lmcache.v1.multiprocess.protocols.base import RequestType
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )
    from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor
    from lmcache.v1.platform.cpu.shm import (
        CpuShmTensorWrapper,
        shm_create_readwrite,
    )
except ImportError as _exc:
    _IMPORT_ERROR = _exc
    # Fallback placeholder so ``add_arguments`` can still build its
    # help text without crashing on a CLI-only install.
    DTYPE_MAP = {}  # type: ignore[assignment]

    # Stubs so other modules (notably ``command.py``) can still import
    # the SHM helpers on a slim install; ``_require_full_install`` is
    # the gate that prevents them from ever being invoked there.
    def shm_open_pool_as_mmap(name: str, nbytes: int) -> Any:  # type: ignore[misc]
        raise RuntimeError(
            "shm_open_pool_as_mmap unavailable on slim lmcache-cli install"
        )


def _require_full_install() -> None:
    """Exit with an install hint if the full LMCache runtime is missing.

    ``lmcache bench server`` needs torch, zmq and ``lmcache.v1.*``
    (MP client, KV layer-group parser). When those imports failed at
    module load — almost always because the user installed
    ``lmcache-cli`` instead of the full package — print the shortest
    actionable message to stderr and exit with status ``2`` so
    scripts can detect the install gap programmatically.
    """
    if _IMPORT_ERROR is None:
        return
    print(
        "ERROR: `lmcache bench server` needs the full LMCache package "
        "(torch, zmq, MP runtime), but only the `lmcache-cli` shell "
        "appears to be installed.\n"
        "  Install the full package with `pip install lmcache` and try "
        "again.\n"
        f"  Original import error: {_IMPORT_ERROR}",
        file=sys.stderr,
    )
    sys.exit(2)


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

_HELLO_TOKEN_ID = 9906
_MODEL_NAME = "test-model"
_WORLD_SIZE = 1
_INSTANCE_ID = 0

# Default KV shape spec matching the original defaults:
# 32 layers, (2, num_blocks=1024, block_size=16, 8 heads, 128 head_size)
_DEFAULT_SHAPE_SPEC = "(2,1024,16,8,128):float16:32"

# Historical prefetch-status poll cadence (seconds). Kept as the default
# for ``--prefetch-poll-interval`` so single-threaded runs are unchanged.
_DEFAULT_PREFETCH_POLL_INTERVAL_S = 0.05

# Base seed for the deterministic KV-cache fills. Worker ``w`` seeds its
# own local generator with ``_BASE_ALLOC_SEED + w`` so every worker's
# band holds *distinct* data — if two contexts ever alias the same
# memory, or a RETRIEVE lands in the wrong worker's band, the checksum
# comparison can actually detect it (identical fills would mask it).
_BASE_ALLOC_SEED = 42


def _no_progress(_msg: str) -> None:
    """Silently drop a per-request progress line.

    Used as the default progress sink for :func:`_process_request` so the
    throughput modes (and ``--quiet``) never print the chatty per-request
    lines that would otherwise distort a high-rate run.

    Args:
        _msg: The progress line, ignored.
    """
    return None


# ------------------------------------------------------------------ #
#  Concurrency / workload enums                                        #
# ------------------------------------------------------------------ #


class OpMode(str, Enum):
    """Workload shape for ``lmcache bench server``.

    Members:
        PAIR: Historical behaviour — a cold STORE pass followed by a
            warm RETRIEVE pass per sequence, with checksum comparison.
        STORE_ONLY: A single cold STORE pass over unique sequences
            (write-throughput measurement).
        RETRIEVE_ONLY: A STORE pre-warm pass followed by a measured warm
            RETRIEVE pass (read-throughput measurement).
    """

    PAIR = "pair"
    STORE_ONLY = "store-only"
    RETRIEVE_ONLY = "retrieve-only"


class ChecksumMode(str, Enum):
    """Whether per-request checksum verification runs.

    Members:
        AUTO: Enable in ``PAIR`` mode, disable in the single-pass
            throughput modes. Resolved to ``ON`` / ``OFF`` by the
            orchestrator before it reaches :func:`_process_request`.
        ON: Always verify checksums.
        OFF: Never verify checksums (throughput path only).
    """

    AUTO = "auto"
    ON = "on"
    OFF = "off"


class RequestContract(str, Enum):
    """Per-request workload contract declared by the driving pass.

    :func:`_process_request` enforces the declared contract after the
    LOOKUP resolves; a violation is reported through
    :attr:`RequestResult.failure` and the operations that would break
    the contract are never issued.

    Members:
        PAIR: Historical mixed flow — RETRIEVE whatever portion hit and
            STORE whatever portion missed. No constraint on the LOOKUP
            outcome itself.
        STORE_MISS: The request must be a full miss
            (``hit_chunks == 0``); the full range is STOREd and no
            RETRIEVE may be issued. Declared by ``--op store-only`` and
            by the ``--op retrieve-only`` pre-warm pass.
        RETRIEVE_HIT: The request must be a full hit
            (``hit_chunks == total_chunks``); the full range is
            RETRIEVEd and no STORE may be issued. Declared by the
            measured pass of ``--op retrieve-only``.
    """

    PAIR = "pair"
    STORE_MISS = "store-miss"
    RETRIEVE_HIT = "retrieve-hit"


def _band_blocks(
    total_blocks: int,
    num_workers: int,
    num_req_blocks: int,
) -> int:
    """Return the per-worker paged-block count (band size).

    Under the one-client-plus-one-context-per-worker topology, each worker
    registers its **own** KV cache of ``total_blocks // num_workers``
    blocks, so the total registered KV tensor capacity never exceeds the
    requested pool and stays approximately constant regardless of the
    concurrency level — the floor division may leave up to
    ``num_workers - 1`` blocks unused (e.g. ``1000 // 3`` gives
    333-block bands, 999 blocks total) — while per-client and
    per-context metadata overhead still scales with the worker count.
    This helper
    computes that band size and fails fast when the pool cannot be
    divided into usable bands.

    With ``num_workers == 1`` the band is the full ``total_blocks`` and no
    size check is applied — the single-worker path keeps the historical
    clamp behaviour (a request larger than the pool clamps to offset 0
    rather than raising), preserving the default-unchanged contract.

    Args:
        total_blocks: Total paged blocks the operator asked for
            (``--num-blocks`` or the shape spec's ``NB``).
        num_workers: Number of concurrent workers (``>= 1``).
        num_req_blocks: Blocks consumed by a single request
            (``num_full_tokens // block_size``).

    Returns:
        The per-worker band size in blocks.

    Raises:
        ValueError: If ``num_workers < 1``, or (for ``num_workers > 1``)
            the per-worker band cannot hold one request's block range.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1, got %d" % num_workers)
    band = total_blocks // num_workers
    if num_workers > 1 and band < num_req_blocks:
        raise ValueError(
            "block partition too small: %d workers over %d blocks gives a "
            "%d-block band, but each request needs %d blocks. Reduce "
            "--concurrency, increase --num-blocks, or shrink --num-tokens."
            % (num_workers, total_blocks, band, num_req_blocks)
        )
    return band


def _request_block_offset(
    seq_no: int,
    num_req_blocks: int,
    band_blocks: int,
) -> int:
    """Compute the paged block offset of one request within a worker's band.

    Each worker owns a private KV cache of ``band_blocks`` blocks (its own
    registered context, addressed from block 0). Successive requests walk a
    different slice of that private buffer so distinct sequences touch
    distinct data:

        ``offset = (seq_no * num_req_blocks) % max(band_blocks - num_req_blocks, 1)``

    Because every worker has its own context, no cross-worker banding is
    needed — the offset is always local to the worker's own tensors. This
    is exactly the historical single-worker formula with
    ``total_blocks`` replaced by the worker's band size, so at
    ``--concurrency 1`` (band == total) it reproduces the legacy layout
    (including the clamp-to-0 behaviour when a request is larger than the
    pool).

    Args:
        seq_no: The request's sequence number (also its cache-key seed).
        num_req_blocks: Blocks consumed by a single request
            (``num_full_tokens // block_size``).
        band_blocks: Total paged blocks in this worker's registered KV
            cache (its band size).

    Returns:
        The starting block index for this request's slice, relative to the
        worker's own tensors.
    """
    usable = max(band_blocks - num_req_blocks, 1)
    return (seq_no * num_req_blocks) % usable


# ------------------------------------------------------------------ #
#  Low-level helpers                                                   #
# ------------------------------------------------------------------ #

# Default RPC call timeout (seconds) for blocking request/reply
# round-trips.
_DEFAULT_RPC_TIMEOUT_S = 10.0

# Unique sentinel returned by :func:`_call` on RPC timeout so callers
# can disambiguate it from a legitimate ``None`` (void) reply.
_TIMEOUT = object()


def _call(
    client: MessageQueueClient,
    request_type: RequestType,
    payloads: list,
    timeout_s: float = _DEFAULT_RPC_TIMEOUT_S,
) -> Any:
    """Submit a request through ``MessageQueueClient`` and block.

    Returns the decoded response (possibly ``None`` for void replies)
    on success, or the sentinel ``_TIMEOUT`` on RPC timeout.
    """
    future: MessagingFuture[Any] = client.submit_request(request_type, payloads)
    try:
        return future.result(timeout=timeout_s)
    except TimeoutError:
        return _TIMEOUT


# ------------------------------------------------------------------ #
#  Token / key helpers                                                 #
# ------------------------------------------------------------------ #


def _build_token_ids(
    seq_no: int,
    num_tokens: int,
) -> tuple[int, ...]:
    """Build token sequence: ``(seq_no, hello, hello, ...)``."""
    return (seq_no,) + (_HELLO_TOKEN_ID,) * num_tokens


def _make_key(
    token_ids: tuple[int, ...],
    request_id: str,
    start: int = 0,
    end: int = 0,
    worker_id: int | None = None,
) -> IPCCacheServerKey:
    """Build an IPCCacheServerKey."""
    return IPCCacheServerKey(
        model_name=_MODEL_NAME,
        world_size=_WORLD_SIZE,
        worker_id=worker_id,
        token_ids=token_ids,
        start=start,
        end=end if end > 0 else len(token_ids),
        request_id=request_id,
    )


# ------------------------------------------------------------------ #
#  Protocol operations                                                 #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  GPU KV cache allocation                                             #
# ------------------------------------------------------------------ #


def _allocate_gpu_kv_cache(
    num_layers: int = 32,
    num_heads: int = 8,
    head_size: int = 128,
    num_blocks: int = 1024,
    block_size: int = 16,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    kv_size: int = 2,
    groups: list[KVLayerGroupInfo] | None = None,
    nb_override: int | None = None,
    seed: int = _BASE_ALLOC_SEED,
) -> list[torch.Tensor]:
    """Allocate paged GPU KV cache tensors.

    Each layer is a tensor of shape
    ``(kv_size, num_blocks, block_size, num_heads, head_size)``
    matching the vLLM NHD layout. ``kv_size`` is 2 for standard
    K/V attention; override via the ``--kvcache-shape-spec``
    first dimension for architectures that need a different
    leading dimension (e.g. MLA).

    When ``groups`` is provided, tensors are allocated per-group
    using each group's own ``(kv_size, NB, BS, NH, HS)`` / ``dtype``
    (for heterogeneous multi-group specs). In that mode the flat
    ``num_heads`` / ``head_size`` / ``dtype`` / ``kv_size`` kwargs
    are ignored, and ``num_layers`` is derived from the groups.

    Args:
        nb_override: When set (``groups`` mode only), allocate this many
            paged blocks per group instead of each group's ``NB``. Used to
            allocate a per-worker band (``total_blocks // concurrency``) so
            the total registered KV tensor capacity does not exceed the
            requested pool and stays approximately constant as
            concurrency grows.
        seed: Seed for the local random generator that fills the
            tensors. Callers pass a distinct per-worker seed
            (``_BASE_ALLOC_SEED + worker_id``) so no two workers hold
            identical KV bytes — a cross-context misread would otherwise
            be undetectable. A local ``torch.Generator`` is used; the
            process-global RNG state is never touched.
    """
    # ``torch.float16`` cannot be used as a default value because the
    # module must load on ``lmcache-cli`` (no torch) installs.
    if dtype is None:
        dtype = torch.float16
    dev = (
        torch.device(device)
        if device
        else torch.device(torch_device_type, torch_dev.current_device())
    )
    # Local generator: deterministic per seed, and it never mutates the
    # process-global RNG (other components' randomness stays untouched).
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    def _alloc(
        shape: tuple[int, ...],
        a_dtype: torch.dtype,
    ) -> torch.Tensor:
        if a_dtype.is_floating_point:
            return torch.randn(shape, dtype=a_dtype, device=dev, generator=gen)
        # ``torch.randn`` only supports floating-point dtypes; fall
        # back to ``randint`` for integer dtypes (e.g. ``uint8``
        # used by FP8 quantized KV cache layouts).
        iinfo = torch.iinfo(a_dtype)
        return torch.randint(
            iinfo.min, iinfo.max + 1, shape, dtype=a_dtype, device=dev, generator=gen
        )

    if groups:
        tensors: list[torch.Tensor] = []
        for g in groups:
            sd = g.shape_desc
            nb = nb_override if nb_override is not None else sd.nb
            g_shape = (sd.kv_size, nb, sd.bs, sd.nh, sd.hs)
            tensors.extend(_alloc(g_shape, g.dtype) for _ in range(sd.nl))
        return tensors

    shape = (kv_size, num_blocks, block_size, num_heads, head_size)
    return [_alloc(shape, dtype) for _ in range(num_layers)]


# Backward-compatible alias used by tests and older callers.
_allocate_kv_cache = _allocate_gpu_kv_cache


def _allocate_cpu_shm_kv_cache(
    groups: list[KVLayerGroupInfo],
    shm_prefix: str,
    nb_override: int | None = None,
    seed: int = _BASE_ALLOC_SEED,
) -> tuple[list[torch.Tensor], list[CpuShmTensorWrapper], list[str]]:
    """Allocate paged CPU KV cache tensors backed by POSIX SHM.

    For each (group, layer) we ``shm_open`` a fresh segment and
    ``mmap`` it into the client process. The returned tensors share
    storage with the SHM mapping, and the matching
    :class:`CpuShmTensorWrapper` instances tell the LMCache mp
    server how to map the very same physical pages -- i.e. true
    zero-copy across processes (matching the GPU CUDA-IPC path).

    Args:
        groups: Parsed KV layer groups describing per-group geometry.
        shm_prefix: Unique per-worker prefix for the SHM segment names so
            concurrent workers never collide on a segment name.
        nb_override: When set, allocate this many paged blocks per group
            instead of each group's ``NB`` (a per-worker band).
        seed: Seed for the local random generator that fills the
            tensors (distinct per worker; see
            :func:`_allocate_gpu_kv_cache`). The process-global RNG
            state is never touched.

    Returns:
        ``(tensors, wrappers, shm_names)``. ``shm_names`` is kept
        so the caller can ``shm_unlink`` on shutdown.
    """
    # Deterministic per-seed fill (reproducible checksums across
    # cold/warm passes) via a local generator: the process-global RNG is
    # never touched, and each worker passes a distinct seed so no two
    # bands hold identical bytes.
    gen = torch.Generator()
    gen.manual_seed(seed)
    tensors: list[torch.Tensor] = []
    wrappers: list[CpuShmTensorWrapper] = []
    shm_names: list[str] = []
    layer_idx = 0
    for g_idx, g in enumerate(groups):
        sd = g.shape_desc
        nb = nb_override if nb_override is not None else sd.nb
        g_shape = (sd.kv_size, nb, sd.bs, sd.nh, sd.hs)
        for _ in range(sd.nl):
            n_elems = 1
            for d in g_shape:
                n_elems *= d
            nbytes = n_elems * g.dtype.itemsize
            name = "%s_%d_%d" % (shm_prefix, g_idx, layer_idx)
            addr = shm_create_readwrite(name, nbytes)
            buf_type = ctypes.c_uint8 * nbytes
            buf = buf_type.from_address(addr)
            flat = torch.frombuffer(buf, dtype=torch.uint8)
            t = flat.view(g.dtype).reshape(g_shape)
            # Initialise with deterministic random data so the
            # cold/warm checksum compare in the bench loop is
            # meaningful.
            if g.dtype.is_floating_point:
                t.copy_(torch.randn(g_shape, dtype=g.dtype, generator=gen))
            else:
                iinfo = torch.iinfo(g.dtype)
                t.copy_(
                    torch.randint(
                        iinfo.min,
                        iinfo.max + 1,
                        g_shape,
                        dtype=g.dtype,
                        generator=gen,
                    )
                )
            tensors.append(t)
            wrappers.append(CpuShmTensorWrapper(t, name))
            shm_names.append(name)
            layer_idx += 1
    return tensors, wrappers, shm_names


def _send_register_kv_cache(
    client: MessageQueueClient,
    instance_id: int = 0,
    model_name: str = _MODEL_NAME,
    world_size: int = _WORLD_SIZE,
    layout_hints: dict | None = None,
    kv_caches: KVCache | None = None,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    engine_group_infos: "list[EngineGroupInfo] | None" = None,
) -> "bool | RegisterEngineDrivenContextResponse":
    """Register a KV cache context with the MP server.

    Dispatches to the correct protocol based on ``use_handle``:

    * Handle mode: ``REGISTER_KV_CACHE`` with a wrapper list
      (``CudaIPCWrapper`` for GPU, ``CpuShmTensorWrapper`` for CPU).
    * Data mode: ``REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` with a
      ``RegisterEngineDrivenContextPayload`` derived from ``layout_hints``.

    ``use_handle`` defaults to ``use_gpu`` for backwards compatibility:
    GPU always goes through the handle path, CPU defaults to data.

    ``engine_group_infos`` (handle mode only) carries the per-group
    metadata — including each group's true ``tokens_per_block`` — so the
    server does not have to trust the block size discovered from the
    tensors (which the HND layout can swap with ``num_heads``). ``None``
    sends an empty list (single non-hybrid group, geometry discovered
    from the tensors).
    """
    if use_handle is None:
        use_handle = use_gpu
    if use_handle:
        if not kv_caches:
            raise ValueError(
                "kv_caches must be a non-empty list of wrappers "
                "(CudaIPCWrapper for GPU, CpuShmTensorWrapper for CPU)"
            )
        hints: dict = {"kv_layout": "NHD"}
        if layout_hints:
            hints.update(layout_hints)
        # TODO(maobaolong): Make the engine type configurable
        payloads = [
            instance_id,
            kv_caches,
            model_name,
            world_size,
            EngineType.VLLM,
            hints,
            list(engine_group_infos or ()),
        ]
        result = _call(client, RequestType.REGISTER_KV_CACHE, payloads)
        return result is not _TIMEOUT

    # CPU mode: use the non-GPU context registration protocol.
    # layout_hints carries num_layers, num_heads, head_size, block_size,
    # dtype.  hidden_dim_size = num_heads * head_size (NHD layout).
    hints_d: dict = layout_hints or {}
    num_layers = int(hints_d.get("num_layers", 32))
    num_heads = hints_d.get("num_heads", 8)
    head_size = hints_d.get("head_size", 128)
    block_size = int(hints_d.get("block_size", 16))
    dtype_str = str(hints_d.get("dtype", "float16"))
    # "mixed" can appear for heterogeneous specs; fall back to first group.
    if not isinstance(num_heads, int):
        num_heads = 8
    if not isinstance(head_size, int):
        head_size = 128
    hidden_dim_size = int(num_heads) * int(head_size)
    payload = RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name=model_name,
        world_size=world_size,
        block_size=block_size,
        num_layers=num_layers,
        hidden_dim_size=hidden_dim_size,
        dtype_str=dtype_str,
        use_mla=False,
    )
    result = _call(
        client, RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT, [payload]
    )
    if result is _TIMEOUT:
        return False
    # The data-mode register reply carries the server's SHM pool name
    # and size; the bench keeps it on the side so STORE / RETRIEVE
    # can mmap the same pool and exchange tensor data without going
    # through pickle.
    return result


def _send_unregister_kv_cache(
    client: MessageQueueClient,
    instance_id: int = 0,
    use_handle: bool = True,
) -> bool:
    """Deregister a KV cache context from the MP server.

    The inverse of :func:`_send_register_kv_cache`. Without this call
    the server keeps the bench's registration (and the CUDA-IPC / POSIX
    SHM mappings it holds) alive forever, leaking one context entry per
    bench run.

    Dispatches to the correct protocol based on ``use_handle``, mirroring
    the register path:

    * Handle mode: ``UNREGISTER_KV_CACHE``.
    * Data mode: ``UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT``.

    Both protocols take a single ``instance_id`` payload and return a void
    reply, so success is distinguished from an RPC timeout only.

    Args:
        client: The MP message-queue client.
        instance_id: The instance ID used at registration time. Must match
            the ``instance_id`` passed to :func:`_send_register_kv_cache`.
        use_handle: ``True`` for the handle path (GPU CUDA-IPC / CPU SHM),
            ``False`` for the engine-driven data path.

    Returns:
        ``True`` if the server acknowledged the call, ``False`` on RPC
        timeout.
    """
    request_type = (
        RequestType.UNREGISTER_KV_CACHE
        if use_handle
        else RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
    )
    result = _call(client, request_type, [instance_id])
    return result is not _TIMEOUT


def _send_unregister_kv_cache_batch(
    clients_and_ids: "list[tuple[MessageQueueClient, int]]",
    use_handle: bool = True,
    timeout_s: float = _DEFAULT_RPC_TIMEOUT_S,
) -> list[bool]:
    """Deregister several contexts concurrently against one shared deadline.

    Every UNREGISTER is submitted up front (non-blocking) and only then
    are the replies awaited, all against a single wall-clock deadline. So
    tearing down ``N`` worker contexts against an unresponsive server
    costs ``~timeout_s`` in total rather than ``timeout_s * N`` — the
    serial submit-then-block pattern of calling
    :func:`_send_unregister_kv_cache` in a loop.

    Args:
        clients_and_ids: ``(client, instance_id)`` pairs to deregister,
            one per worker context (each registered under its own client
            identity).
        use_handle: ``True`` for the handle protocol (GPU CUDA-IPC / CPU
            SHM), ``False`` for the engine-driven data path. Uniform
            across a run.
        timeout_s: Shared deadline for the whole batch, in seconds.

    Returns:
        Per-entry acknowledgement flags in input order: ``True`` if the
        server replied before the shared deadline, ``False`` on timeout.
    """
    request_type = (
        RequestType.UNREGISTER_KV_CACHE
        if use_handle
        else RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
    )
    futures: list[MessagingFuture[Any]] = [
        client.submit_request(request_type, [instance_id])
        for client, instance_id in clients_and_ids
    ]
    deadline = time.monotonic() + timeout_s
    results: list[bool] = []
    for fut in futures:
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            fut.result(timeout=remaining)
            results.append(True)
        except TimeoutError:
            results.append(False)
    return results


def _send_lookup(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
) -> bool:
    """LOOKUP — submit a prefix lookup.

    The server-side handler returns ``None`` (void) on success, so
    we only distinguish RPC timeout from a completed call.
    """
    result = _call(client, RequestType.LOOKUP, [key, 1])
    return result is not _TIMEOUT


def _poll_prefetch_status(
    client: MessageQueueClient,
    request_id: str,
    max_polls: int = 50,
    poll_interval: float = 0.05,
) -> int | None:
    """QUERY_PREFETCH_STATUS — poll until done.

    Returns the hit chunk count, or ``None`` if the polling budget
    is exhausted. The server keys prefetch jobs by ``request_id``
    (str), not an integer job handle.
    """
    for _ in range(max_polls):
        result = _call(
            client,
            RequestType.QUERY_PREFETCH_STATUS,
            [request_id],
        )
        if result is _TIMEOUT:
            # RPC timeout — treat as giving up on this poll cycle.
            return None
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _make_event_handle(use_gpu: bool = True) -> bytes:
    """Create a CUDA event IPC handle for GPU mode.

    CPU mode does not need a cross-process event (SHM mappings are
    coherent without device-side sync), so an empty handle is
    returned and the server treats it as a no-op.
    """
    if not use_gpu:
        return b""
    check_interprocess_event_support()
    event = torch_dev.Event(interprocess=True)
    event.record()
    return event.ipc_handle()


def _build_server_slot_views(
    server_pool: "mmap.mmap",
    slots: list[dict[str, Any]],
) -> list["torch.Tensor"]:
    """Build zero-copy tensor views over server SHM slot descriptors.

    Each ``ShmSlotDescriptor`` carries the ``(offset, length, shape,
    dtype)`` of one chunk inside the server-owned SHM pool; we wrap
    them with ``torch.frombuffer`` so the bench can read or overwrite
    that chunk without going through pickle.
    """
    views: list[torch.Tensor] = []
    for raw in slots:
        desc = ShmSlotDescriptor.from_dict(raw)
        dtype = getattr(torch, desc.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("invalid torch dtype string: %s" % desc.dtype)
        itemsize = torch.empty((), dtype=dtype).element_size()
        if itemsize <= 0:
            raise ValueError("invalid dtype size for %s" % desc.dtype)
        count = desc.length // itemsize
        flat = torch.frombuffer(
            server_pool, dtype=dtype, count=count, offset=desc.offset
        )
        views.append(flat.view(torch.Size(desc.shape)))
    return views


def _gather_paged_to_flat_chunks(
    tensors: list["torch.Tensor"],
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
) -> list["torch.Tensor"]:
    """Gather paged client tensors into flat per-chunk CPU tensors.

    Output layout matches the server's expected ``commit_store``
    payload (set up at register time by
    ``register_kv_cache_engine_driven_context``):
    each chunk is ``[2, num_layers, chunk_size, hidden_dim]``,
    where ``hidden_dim = NH * HS``. Assumes a homogeneous group
    (same NH/HS/dtype across all layers); heterogeneous specs
    fall outside the bench scope.
    """
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size %d must be a multiple of block_size %d"
            % (chunk_size, block_size)
        )
    blocks_per_chunk = chunk_size // block_size
    num_chunks = num_blocks // blocks_per_chunk
    num_layers = len(tensors)
    chunks: list[torch.Tensor] = []
    for c in range(num_chunks):
        start_b = block_offset + c * blocks_per_chunk
        per_layer: list[torch.Tensor] = []
        for t in tensors:
            # paged: (2, NB, BS, NH, HS) -> slice block range ->
            # (2, blocks_per_chunk, BS, NH, HS) -> flatten to
            # (2, chunk_size, NH*HS).
            sliced = t.narrow(1, start_b, blocks_per_chunk)
            kv, _, bs, nh, hs = sliced.shape
            flat = sliced.contiguous().view(kv, blocks_per_chunk * bs, nh * hs)
            per_layer.append(flat)
        # Stack along a new layer dim -> (2, NL, chunk_size, hidden).
        chunk = torch.stack(per_layer, dim=1).contiguous()
        if chunk.shape[1] != num_layers:
            raise RuntimeError(
                "unexpected chunk shape %s (NL mismatch)" % (chunk.shape,)
            )
        chunks.append(chunk)
    return chunks


def _scatter_flat_chunks_to_paged(
    tensors: list["torch.Tensor"],
    chunks: list["torch.Tensor"],
    block_offset: int,
    block_size: int,
    chunk_size: int,
) -> None:
    """Inverse of :func:`_gather_paged_to_flat_chunks`.

    Writes each ``[2, NL, chunk_size, hidden]`` flat chunk back into
    the paged client tensors at the matching block range. Used by
    the data-mode RETRIEVE path so the bench's client-side checksum
    can compare cold ground truth with what the server returned.
    """
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size %d must be a multiple of block_size %d"
            % (chunk_size, block_size)
        )
    blocks_per_chunk = chunk_size // block_size
    for c, chunk in enumerate(chunks):
        start_b = block_offset + c * blocks_per_chunk
        for layer_idx, t in enumerate(tensors):
            kv, _, bs, nh, hs = t.shape
            target = t.narrow(1, start_b, blocks_per_chunk)
            # chunk[:, layer_idx] is (chunk_size, hidden); reshape
            # back to (2, blocks_per_chunk, BS, NH, HS).
            flat = chunk[:, layer_idx]
            reshaped = flat.reshape(kv, blocks_per_chunk, bs, nh, hs)
            target.copy_(reshaped)


# ------------------------------------------------------------------ #
#  Client-side checksum / zero-fill (data-mode self-check)             #
# ------------------------------------------------------------------ #


def _compute_client_checksums(
    tensors: list["torch.Tensor"],
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
) -> list[str]:
    """Hash a paged block range from client-side KV tensors.

    For each chunk (``chunk_size // block_size`` consecutive blocks),
    feed every layer's bytes for that block range into a single MD5
    digest. The returned list maps 1:1 to the chunks the bench loop
    expects, so a cold-pass digest can be compared with a warm-pass
    digest to verify that ``RETRIEVE`` actually wrote back the data
    we wrote during ``STORE`` -- without relying on a server-side
    ``/cache/checksums`` endpoint (which only exists in handle mode).
    """
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size %d must be a multiple of block_size %d"
            % (chunk_size, block_size)
        )
    blocks_per_chunk = chunk_size // block_size
    num_chunks = num_blocks // blocks_per_chunk
    checksums: list[str] = []
    for c in range(num_chunks):
        start_b = block_offset + c * blocks_per_chunk
        end_b = start_b + blocks_per_chunk
        h = hashlib.md5()
        for t in tensors:
            # Paged layout: dim 1 is the block dim for both kv-major
            # ``(kv, NB, BS, NH, HS)`` and MLA ``(NB, BS, NH, HS)``
            # tensors. ``contiguous().numpy().tobytes()`` survives
            # non-contiguous slices and dtype quirks (bfloat16 has no
            # numpy view, but uint8 reinterpret works after slice).
            view = t.narrow(1, start_b, end_b - start_b).contiguous()
            if view.device.type != "cpu":
                # ``numpy()`` needs host memory; device tensors (GPU
                # engine_driven runs) are copied down first.
                view = view.cpu()
            h.update(view.view(torch.uint8).numpy().tobytes())
        checksums.append(h.hexdigest())
    return checksums


def _zero_fill_client_blocks(
    tensors: list["torch.Tensor"],
    block_offset: int,
    num_blocks: int,
) -> None:
    """Zero out a paged block range across all client tensors.

    Used right before a warm-pass ``RETRIEVE`` so that any non-zero
    bytes observed afterwards must have been written by the server.
    Without this, a warm checksum equal to the cold checksum could
    still happen even if ``RETRIEVE`` was a silent no-op (the SHM
    pages were never overwritten in the first place).
    """
    for t in tensors:
        t.narrow(1, block_offset, num_blocks).zero_()


def _send_store(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    block_offset: int = 0,
    block_size: int = 16,
    num_engine_group_infos: int = 1,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    client_tensors: list["torch.Tensor"] | None = None,
    chunk_size: int = 0,
    server_pool: "mmap.mmap | None" = None,
    instance_id: int = _INSTANCE_ID,
) -> str:
    """Store KV cache blocks. Returns status string.

    Handle mode uses the single-shot ``STORE`` RPC (GPU CUDA-IPC, or
    CPU SHM with an empty event handle).
    Data mode uses the two-phase ``PREPARE_STORE`` + ``COMMIT_STORE``.
    When ``server_pool`` and ``client_tensors`` are both supplied the
    bench gathers the paged block range into flat per-chunk CPU
    tensors and writes them straight into the server-owned SHM pool
    via the slot descriptors returned by ``PREPARE_STORE``, so the
    follow-up ``COMMIT_STORE`` carries an empty payload and the
    server stays on its zero-copy SHM path.

    Args:
        instance_id: The registered context ID to store against. Under
            concurrency each worker owns its own context, so this must be
            the calling worker's instance ID (default ``0`` for the
            single-worker path).

    Returns:
        ``"stored"`` on success, ``"store_failed"`` if the server
        declined the store, or ``"timeout"`` on RPC timeout.
    """
    if use_handle is None:
        use_handle = use_gpu
    if use_handle:
        num_tokens = key.end - key.start
        num_blocks = num_tokens // block_size
        block_ids = list(range(block_offset, block_offset + num_blocks))
        payloads = [
            key,
            instance_id,
            [block_ids] * num_engine_group_infos,
            # ``use_handle`` also covers cpu lmcache_driven: pass the
            # mode through so CPU runs send the empty event handle
            # instead of incorrectly building a CUDA event.
            _make_event_handle(use_gpu),
        ]
        result = _call(client, RequestType.STORE, payloads)
        if result is _TIMEOUT:
            return "timeout"
        return "stored" if result[1] else "store_failed"

    # CPU mode: PREPARE_STORE -> COMMIT_STORE
    prep = _call(client, RequestType.PREPARE_STORE, [key, instance_id])
    if prep is _TIMEOUT:
        return "timeout"
    if server_pool is not None and client_tensors is not None and chunk_size > 0:
        ctx = prep.context if isinstance(prep.context, dict) else {}
        slots = ctx.get("slots", []) or []
        chunk_indices = ctx.get("chunk_indices", []) or []
        if slots and chunk_indices:
            num_blocks = (key.end - key.start) // block_size
            full_chunks = _gather_paged_to_flat_chunks(
                client_tensors,
                block_offset,
                num_blocks,
                block_size,
                chunk_size,
            )
            slot_views = _build_server_slot_views(server_pool, slots)
            for slot_view, chunk_idx in zip(slot_views, chunk_indices, strict=False):
                if 0 <= chunk_idx < len(full_chunks):
                    slot_view.copy_(full_chunks[chunk_idx].view(slot_view.shape))
    commit = _call(client, RequestType.COMMIT_STORE, [key, instance_id, b""])
    if commit is _TIMEOUT:
        return "timeout"
    return "stored" if commit else "store_failed"


def _send_retrieve(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    chunk_size: int,
    hit_chunks: int,
    block_offset: int = 0,
    block_size: int = 16,
    num_engine_group_infos: int = 1,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    client_tensors: list["torch.Tensor"] | None = None,
    server_pool: "mmap.mmap | None" = None,
    instance_id: int = _INSTANCE_ID,
) -> str:
    """Retrieve KV cache blocks. Returns status.

    Handle mode uses the single-shot ``RETRIEVE`` RPC (GPU CUDA-IPC, or
    CPU SHM with an empty event handle).
    Data mode uses the two-phase ``PREPARE_RETRIEVE`` +
    ``COMMIT_RETRIEVE``. When ``server_pool`` and ``client_tensors``
    are both supplied the bench builds zero-copy tensor views over
    the slot descriptors returned by ``PREPARE_RETRIEVE`` and
    scatters them back into the paged client SHM, so the round-trip
    self-check can run without ``PREPARE_RETRIEVE`` having to ship a
    pickled copy of the chunks.

    Args:
        instance_id: The registered context ID to retrieve from. Under
            concurrency each worker owns its own context, so this must be
            the calling worker's instance ID (default ``0`` for the
            single-worker path).

    Returns:
        ``"retrieved"`` on success, ``"retrieve_failed"`` if the server
        reported a miss / declined, or ``"timeout"`` on RPC timeout.
    """
    if use_handle is None:
        use_handle = use_gpu
    if use_handle:
        hit_tokens = hit_chunks * chunk_size
        num_blocks = hit_tokens // block_size
        block_ids = list(range(block_offset, block_offset + num_blocks))
        payloads = [
            key,
            instance_id,
            [block_ids] * num_engine_group_infos,
            # CPU handle mode sends the empty event handle (see
            # _send_store); only GPU builds a CUDA IPC event.
            _make_event_handle(use_gpu),
            0,  # skip_first_n_tokens
        ]
        result = _call(client, RequestType.RETRIEVE, payloads)
        if result is _TIMEOUT:
            return "timeout"
        return "retrieved" if result[1] else "retrieve_failed"

    # CPU mode: PREPARE_RETRIEVE -> COMMIT_RETRIEVE
    prep = _call(client, RequestType.PREPARE_RETRIEVE, [key, instance_id])
    if prep is _TIMEOUT:
        return "timeout"
    if not prep.success:
        return "retrieve_failed"
    if server_pool is not None and client_tensors is not None:
        ctx = prep.context if isinstance(prep.context, dict) else {}
        slots = ctx.get("slots", []) or []
        if slots:
            try:
                slot_views = _build_server_slot_views(server_pool, slots)
                _scatter_flat_chunks_to_paged(
                    client_tensors,
                    slot_views,
                    block_offset,
                    block_size,
                    chunk_size,
                )
            except (RuntimeError, ValueError) as exc:
                print(
                    "  [WARNING] retrieve scatter failed: %s" % exc,
                    file=sys.stderr,
                )
    commit = _call(client, RequestType.COMMIT_RETRIEVE, [key, instance_id])
    if commit is _TIMEOUT:
        return "timeout"
    return "retrieved" if commit else "retrieve_failed"


def _send_end_session(
    client: MessageQueueClient,
    request_id: str,
) -> bool:
    """END_SESSION — clean up server-side session state.

    Args:
        client: The MP message-queue client.
        request_id: The request whose session should be removed.

    Returns:
        ``True`` if the server acknowledged the call, ``False`` on RPC
        timeout. A timeout means the session may still be held
        server-side, so the caller must fail the run and treat the
        server as tainted rather than silently reporting success.
    """
    return _call(client, RequestType.END_SESSION, [request_id]) is not _TIMEOUT


def _send_free_lookup_locks(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
) -> bool:
    """FREE_LOOKUP_LOCKS — release read locks acquired by a prior LOOKUP.

    A successful LOOKUP takes a read lock on every hit chunk so the
    engine can retrieve it without it being evicted underneath. The
    normal RETRIEVE flow consumes those locks, but a request that bails
    after the LOOKUP without retrieving (a workload-contract violation on
    a hit) would otherwise leak them. Call this with a key covering the
    hit range so the server releases exactly the locks it acquired,
    mirroring the ``tp_size`` argument used by :func:`_send_lookup`.

    Args:
        client: The MP message-queue client.
        key: A cache key whose ``[start, end)`` range covers the chunks
            whose read locks should be released (the hit range).

    Returns:
        ``True`` if the server acknowledged the call, ``False`` on RPC
        timeout. A timeout means the read locks may still be held, so
        the caller must fail the run and treat the server as tainted.
    """
    return _call(client, RequestType.FREE_LOOKUP_LOCKS, [key, 1]) is not _TIMEOUT


# ------------------------------------------------------------------ #
#  Checksum query                                                      #
# ------------------------------------------------------------------ #


def _query_checksum(
    http_base: str,
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
    instance_id: int = _INSTANCE_ID,
) -> list[str] | None:
    """Query KV cache checksums via the HTTP API.

    Uses the MP-native ``block_ids`` + ``block_size`` addressing
    scheme so the query matches the same block-level semantics
    as ``STORE`` / ``RETRIEVE``. This CLI pins ``layerwise=false``
    so the server always returns ``chunk_checksums`` as a flat
    ``list[str]``. We still defensively validate the response
    type — if a future endpoint variant returns a per-layer
    ``dict`` we log and skip the comparison rather than letting
    ``str.join`` crash.

    Args:
        instance_id: The registered context to hash. The
            ``/cache/checksums`` endpoint is instance-scoped, so under
            concurrency this must be the calling worker's instance ID;
            otherwise every worker would hash instance 0's tensors.
    """
    blocks = list(range(block_offset, block_offset + num_blocks))
    # The MP /cache/checksums endpoint is block-native: its chunk_size counts
    # blocks per chunk, while our caller passes in the server-side token-level
    # chunk_size. Convert here.
    if chunk_size % block_size != 0:
        print(
            "  [WARNING] chunk_size %d not a multiple of block_size %d; "
            "skipping checksum query" % (chunk_size, block_size),
            file=sys.stderr,
        )
        return None
    chunk_size_blocks = chunk_size // block_size
    url = "%s/cache/checksums" % http_base
    payload = json.dumps(
        {
            "block_ids": blocks,
            "chunk_size": chunk_size_blocks,
            "instance_id": instance_id,
            "layerwise": False,
        }
    ).encode()
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") != "success":
                return None
            checksums = data.get("chunk_checksums", [])
            if not isinstance(checksums, list) or not all(
                isinstance(c, str) for c in checksums
            ):
                print(
                    "  [WARNING] unexpected chunk_checksums "
                    "type=%s; expected list[str]" % type(checksums).__name__,
                    file=sys.stderr,
                )
                return None
            return checksums
    except (urllib.error.URLError, OSError) as exc:
        print("  [WARNING] Checksum query failed: %s" % exc, file=sys.stderr)
    return None


# ------------------------------------------------------------------ #
#  Per-request flow                                                    #
# ------------------------------------------------------------------ #


class ServerTaintedInterrupt(KeyboardInterrupt):
    """A Ctrl-C whose request cleanup (END_SESSION) also timed out.

    A :class:`KeyboardInterrupt` subclass so it keeps interrupt semantics
    (the run still exits 130), while signalling that the server may hold
    residual state and must be restarted. Raised from the
    :func:`_process_request` finally when the interrupted request could not
    acknowledge END_SESSION; the orchestrator detects it and reports
    ``Server reuse safe: no``.
    """


class ServerTaintedError(RuntimeError):
    """A request failed *and* its cleanup (END_SESSION) also timed out.

    Raised from :func:`_process_request` only on the exception path — when
    the request body raised before a :class:`RequestResult` existed and the
    subsequent END_SESSION could not be acknowledged. It carries the taint
    out so the driver can both invalidate the run and flag the server for a
    restart; the original body exception is preserved via ``raise ... from``.
    """


@dataclass
class RequestResult:
    """Result of a single request pass (cold or warm).

    Carries the checksum list (for correctness verification), per-operation
    latency measurements (for metrics aggregation), and the store /
    retrieve completion status (for failure / timeout accounting).

    Args:
        store_status: Outcome of the STORE this pass attempted, one of
            ``"stored"`` / ``"store_failed"`` / ``"timeout"``, or ``None``
            when no STORE was attempted (a full cache hit).
        retrieve_status: Outcome of the RETRIEVE this pass attempted, one
            of ``"retrieved"`` / ``"retrieve_failed"`` / ``"timeout"``, or
            ``None`` when no RETRIEVE was attempted (a full cache miss).
        store_tokens: Tokens successfully stored (``0`` unless
            ``store_status == "stored"``); feeds the store MB/s metric.
        retrieve_tokens: Tokens successfully retrieved (``0`` unless
            ``retrieve_status == "retrieved"``); feeds the retrieve MB/s
            metric.
        failure: Non-empty human-readable reason when this request must
            invalidate the run: a LOOKUP timeout, a prefetch-status poll
            failure, a STORE / RETRIEVE failure or timeout, or a
            violation of the declared :class:`RequestContract`. The
            statuses / counters above are still populated for
            diagnostics; the driver aborts the worker on any non-empty
            value.
        server_tainted: ``True`` when this request may have left
            unrecoverable state on the server that a nonce cannot avoid:
            a prefetch-status poll that timed out (the server keeps the
            prefetch job — END_SESSION does not cancel it), or a cleanup
            RPC (END_SESSION / FREE_LOOKUP_LOCKS) that timed out (session
            or read locks may still be held). The orchestrator surfaces
            this as ``server_reuse_safe: no`` and the dedicated server
            must be restarted before the next run.
    """

    checksums: list[str] | None = None
    lookup_ms: float | None = None
    retrieve_ms: float | None = None
    store_ms: float | None = None
    hit_chunks: int = 0
    total_chunks: int = 0
    store_tokens: int = 0
    retrieve_tokens: int = 0
    store_status: str | None = None
    retrieve_status: str | None = None
    failure: str = ""
    server_tainted: bool = False


def _process_request(
    client: MessageQueueClient,
    seq_no: int,
    num_tokens: int,
    chunk_size: int,
    pass_label: str,
    http_base: str = "",
    block_size: int = 16,
    total_blocks: int = 1024,
    num_engine_group_infos: int = 1,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    client_tensors: list["torch.Tensor"] | None = None,
    server_pool: "mmap.mmap | None" = None,
    worker_id: int = 0,
    instance_id: int = 0,
    nonce: int = 0,
    poll_interval: float = _DEFAULT_PREFETCH_POLL_INTERVAL_S,
    checksum: ChecksumMode = ChecksumMode.ON,
    contract: RequestContract = RequestContract.PAIR,
    handle_tensors: list["torch.Tensor"] | None = None,
    progress: "Callable[[str], None] | None" = None,
) -> RequestResult | None:
    """Run the full lookup -> retrieve/store flow.

    When ``client_tensors`` is provided (data-mode self-check) and
    ``checksum`` is not :attr:`ChecksumMode.OFF`, the flow gains two
    extra steps:

    * cold pass: hash the paged block range *before* ``STORE``, so
      the digest captures the ground-truth KV bytes.
    * warm pass: zero-fill the same block range *before*
      ``RETRIEVE``, then hash *after* ``RETRIEVE``. cold == warm
      proves the server returned the exact bytes we sent.

    Handle mode keeps the historical server-side ``/cache/checksums``
    digest, but the warm pass zero-fills the shared SHM / IPC pages
    (via ``handle_tensors``) before ``RETRIEVE``: because client and
    server hash the same physical pages, the post-retrieve digest can
    only match the cold digest if the server actually wrote the bytes
    back — a silent no-op RETRIEVE fails instead of trivially passing.

    Each worker drives its **own** registered KV cache of
    ``total_blocks`` blocks (its band), so the block offset is computed
    locally against that band via :func:`_request_block_offset` — no
    cross-worker banding is needed because two workers never share a
    context.

    Args:
        worker_id: Zero-based index of the calling worker. Used only for
            progress-line labelling; the block layout is local to this
            worker's own registered KV cache.
        instance_id: The registered context ID this worker owns. Forwarded
            to STORE / RETRIEVE so each worker addresses its own context.
        nonce: Per-run identity prefix mixed into ``request_id`` (the same
            run nonce that seeds ``instance_id``). Distinct nonces across
            runs keep ``request_id`` values from colliding on the server's
            session table between successive bench runs against one server.
        total_blocks: Paged blocks in *this worker's* registered KV cache
            (its band size). At ``--concurrency 1`` this equals the full
            pool, reproducing the historical block layout exactly.
        poll_interval: Seconds between ``QUERY_PREFETCH_STATUS`` polls
            while waiting for a warm LOOKUP to become ready.
        checksum: Whether to run the per-request checksum verification.
            :attr:`ChecksumMode.OFF` skips the ground-truth capture,
            zero-fill, and digest steps so a throughput run is not
            skewed by the checksum round-trip. :attr:`ChecksumMode.AUTO`
            must be resolved to ``ON`` / ``OFF`` by the caller before
            this function is reached.
        contract: The workload contract this request must satisfy (see
            :class:`RequestContract`). Violations set
            :attr:`RequestResult.failure` and suppress the operation
            that would break the contract.
        handle_tensors: Handle-mode only — the client-side references to
            the registered KV tensors (CUDA IPC / POSIX SHM pages shared
            with the server). Used to zero-fill the target block range
            before a warm RETRIEVE so the server-side checksum becomes a
            discriminating oracle; ``None`` in data mode (which uses
            ``client_tensors`` and client-side hashing instead).
        progress: Optional per-request progress sink. ``None`` (the
            default, used by the throughput modes and ``--quiet``)
            suppresses the chatty LOOKUP / STORE / RETRIEVE lines so they
            do not distort a high-rate run; ``pair`` mode passes the
            run's logger. Warnings and skips always go to ``stderr``
            regardless of this sink.

    Returns:
        A :class:`RequestResult` with the pass's latencies, checksums,
        statuses, and (when the request must invalidate the run) a
        non-empty ``failure``; or ``None`` only when the request was
        skipped because it carries fewer tokens than one chunk.
    """
    emit: "Callable[[str], None]" = progress if progress is not None else _no_progress
    verify_checksum = checksum is not ChecksumMode.OFF
    token_ids = _build_token_ids(seq_no, num_tokens)
    request_id = "req-%d-%d-%s" % (nonce, seq_no, pass_label)

    # Align end to chunk_size (only full chunks)
    num_full_tokens = (len(token_ids) // chunk_size) * chunk_size
    if num_full_tokens == 0:
        print(
            "  [seq %d/%s] SKIP: %d tokens < chunk_size %d"
            % (seq_no, pass_label, len(token_ids), chunk_size),
            file=sys.stderr,
        )
        return None

    # Key for lookup (worker_id=None)
    lookup_key = _make_key(
        token_ids,
        request_id,
        start=0,
        end=num_full_tokens,
    )

    total_chunks = num_full_tokens // chunk_size

    # Holds whatever RequestResult is about to be returned from inside the
    # try so the finally can taint it if END_SESSION fails. Kept as a local
    # (not a return value) because the finally runs on every exit path.
    result: "RequestResult | None" = None
    t0 = time.monotonic()
    try:
        # 1. LOOKUP. Once the LOOKUP is *submitted* the server may have
        # accepted it and created session / prefetch state we cannot
        # observe, so a LOOKUP timeout is a server-effect-unknown failure:
        # fail AND taint. The finally still best-effort END_SESSIONs (a
        # no-op if the server never saw the request), and any Ctrl-C from
        # here on is covered by the same finally.
        if not _send_lookup(client, lookup_key):
            print(
                "  [seq %d/%s] LOOKUP timeout" % (seq_no, pass_label),
                file=sys.stderr,
            )
            result = RequestResult(
                total_chunks=total_chunks,
                failure="LOOKUP timeout (seq %d, %s pass); server effect unknown"
                % (seq_no, pass_label),
                server_tainted=True,
            )
            return result

        # 2. QUERY_PREFETCH_STATUS (poll by request_id). A poll failure —
        # RPC timeout or exhausted poll budget — is a failure, never a miss:
        # converting it to ``hit_chunks = 0`` would make a store-only /
        # pre-warm pass silently re-STORE (and a pair warm pass skip its
        # RETRIEVE) on top of a broken server.
        hit_chunks = _poll_prefetch_status(
            client, lookup_key.request_id, poll_interval=poll_interval
        )
        if hit_chunks is None:
            print(
                "  [seq %d/%s] prefetch status poll failed "
                "(RPC timeout or poll budget exhausted)" % (seq_no, pass_label),
                file=sys.stderr,
            )
            # On a poll timeout the server keeps the prefetch job
            # (END_SESSION removes only the session, not the job), so the
            # dedicated server is tainted and must not be reused.
            result = RequestResult(
                total_chunks=total_chunks,
                failure="prefetch status poll failed (seq %d, %s pass)"
                % (seq_no, pass_label),
                server_tainted=True,
            )
            return result

        miss_chunks = total_chunks - hit_chunks
        hit_tokens = hit_chunks * chunk_size
        lookup_ms = (time.monotonic() - t0) * 1000

        emit(
            "  [w%d seq %d/%s] LOOKUP: %d/%d chunks hit "
            "(%.1f ms)"
            % (
                worker_id,
                seq_no,
                pass_label,
                hit_chunks,
                total_chunks,
                lookup_ms,
            )
        )

        # Enforce the declared workload contract on the LOOKUP outcome
        # *before* any transfer is issued: a store-only / pre-warm request
        # must never RETRIEVE, and a measured retrieve-only request must
        # never STORE — otherwise the reported throughput silently measures
        # the wrong operation mix.
        if contract is RequestContract.STORE_MISS and hit_chunks != 0:
            # LOOKUP took read locks on the hit chunks; release them
            # before bailing so a violated pre-warm never leaks locks. A
            # free-locks timeout means the locks may still be held -> taint.
            freed = _send_free_lookup_locks(
                client, _make_key(token_ids, request_id, start=0, end=hit_tokens)
            )
            result = RequestResult(
                lookup_ms=lookup_ms,
                hit_chunks=hit_chunks,
                total_chunks=total_chunks,
                failure=(
                    "workload contract violation: expected a full miss but "
                    "%d/%d chunks hit (seq %d already cached? use a fresh "
                    "--start range or restart the server)"
                    % (hit_chunks, total_chunks, seq_no)
                ),
                server_tainted=not freed,
            )
            return result
        if contract is RequestContract.RETRIEVE_HIT and hit_chunks != total_chunks:
            # A partial hit still took read locks on the hit chunks;
            # release them before bailing (a full miss took none). A
            # free-locks timeout means the locks may still be held -> taint.
            freed = True
            if hit_chunks > 0:
                freed = _send_free_lookup_locks(
                    client,
                    _make_key(token_ids, request_id, start=0, end=hit_tokens),
                )
            result = RequestResult(
                lookup_ms=lookup_ms,
                hit_chunks=hit_chunks,
                total_chunks=total_chunks,
                server_tainted=not freed,
                failure=(
                    "workload contract violation: expected a full hit but "
                    "only %d/%d chunks hit (seq %d; pre-warm incomplete or "
                    "entries evicted)" % (hit_chunks, total_chunks, seq_no)
                ),
            )
            return result

        # Block offset: each request uses a different block range so distinct
        # sequences touch distinct data. The offset is local to this worker's
        # own registered KV cache (band), so two workers never write the same
        # slice; the range [block_offset, block_offset + num_blocks) always
        # stays inside this worker's own tensors.
        num_blocks = num_full_tokens // block_size
        block_offset = _request_block_offset(
            seq_no,
            num_blocks,
            total_blocks,
        )

        # Client-side self-check (data mode only). cold pass: snapshot
        # ground truth before STORE. warm pass: zero out the slice so
        # a successful RETRIEVE must overwrite every byte. Skipped entirely
        # when checksum verification is off (throughput path).
        cold_ground_truth: list[str] | None = None
        if verify_checksum and client_tensors is not None:
            if pass_label == "cold" and miss_chunks > 0:
                store_block_off = block_offset + (hit_tokens // block_size)
                store_num_blocks = (num_full_tokens - hit_tokens) // block_size
                cold_ground_truth = _compute_client_checksums(
                    client_tensors,
                    store_block_off,
                    store_num_blocks,
                    block_size,
                    chunk_size,
                )
            if pass_label == "warm" and hit_chunks > 0:
                retr_num_blocks = hit_tokens // block_size
                _zero_fill_client_blocks(
                    client_tensors,
                    block_offset,
                    retr_num_blocks,
                )
        elif verify_checksum and handle_tensors is not None:
            # Handle mode: the client and the server share the same physical
            # pages (CUDA IPC / POSIX SHM), so without destroying the pages
            # first, the warm /cache/checksums digest would equal the cold
            # one even if RETRIEVE were a silent no-op — the pages were
            # never disturbed. Zero-fill the target block range so the only
            # way the warm digest can match the cold one is the server
            # actually writing the bytes back.
            if pass_label == "warm" and hit_chunks > 0:
                retr_num_blocks = hit_tokens // block_size
                _zero_fill_client_blocks(
                    handle_tensors,
                    block_offset,
                    retr_num_blocks,
                )
                if use_gpu:
                    # CUDA ordering: the zero-fill kernels were enqueued on
                    # this process's current stream, while the RETRIEVE
                    # event handle is recorded inside ``_send_retrieve``.
                    # Synchronize so the poison is fully committed to the
                    # shared pages before the event is recorded / the RPC
                    # is sent — otherwise the server's write-back could
                    # race the zero-fill and be partially overwritten by it.
                    torch_dev.synchronize()

        # 3. RETRIEVE hit portion
        retrieve_ms: float = 0.0
        store_ms: float = 0.0
        retrieve_status: str | None = None
        store_status: str | None = None
        retrieve_tokens = 0
        store_tokens = 0
        failure = ""
        if hit_chunks > 0:
            retrieve_key = _make_key(
                token_ids,
                request_id,
                start=0,
                end=hit_tokens,
                worker_id=0,
            )
            t1 = time.monotonic()
            retrieve_status = _send_retrieve(
                client,
                retrieve_key,
                chunk_size,
                hit_chunks,
                block_offset=block_offset,
                block_size=block_size,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=client_tensors,
                server_pool=server_pool,
                instance_id=instance_id,
            )
            retrieve_ms = (time.monotonic() - t1) * 1000
            # Count tokens only on a genuine success so MB/s reflects data the
            # server actually returned, not attempts that failed or timed out.
            if retrieve_status == "retrieved":
                retrieve_tokens = hit_tokens
            emit(
                "  [w%d seq %d/%s] RETRIEVE: %s "
                "(%d tokens, %.1f ms)"
                % (
                    worker_id,
                    seq_no,
                    pass_label,
                    retrieve_status,
                    hit_tokens,
                    retrieve_ms,
                )
            )
            if retrieve_status != "retrieved":
                failure = "RETRIEVE %s (seq %d, %s pass)" % (
                    retrieve_status,
                    seq_no,
                    pass_label,
                )

        # 4. STORE miss portion (skipped once the pass already failed).
        if not failure and miss_chunks > 0:
            store_start = hit_tokens
            store_end = num_full_tokens
            store_key = _make_key(
                token_ids,
                request_id,
                start=store_start,
                end=store_end,
                worker_id=0,
            )
            t2 = time.monotonic()
            store_block_off = block_offset + (hit_tokens // block_size)
            store_status = _send_store(
                client,
                store_key,
                block_offset=store_block_off,
                block_size=block_size,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=client_tensors,
                chunk_size=chunk_size,
                server_pool=server_pool,
                instance_id=instance_id,
            )
            store_ms = (time.monotonic() - t2) * 1000
            if store_status == "stored":
                store_tokens = store_end - store_start
            emit(
                "  [w%d seq %d/%s] STORE: %s "
                "(%d tokens, %.1f ms)"
                % (
                    worker_id,
                    seq_no,
                    pass_label,
                    store_status,
                    store_end - store_start,
                    store_ms,
                )
            )
            if store_status != "stored":
                failure = "STORE %s (seq %d, %s pass)" % (
                    store_status,
                    seq_no,
                    pass_label,
                )

        # 5. Compute checksums.
        #   * data mode (client_tensors set):
        #       cold -> ground truth captured pre-STORE
        #       warm -> hash post-RETRIEVE; cold == warm proves the
        #               server returned the exact bytes we wrote.
        #   * handle mode: query /cache/checksums on the server, which
        #     reads the shared SHM/IPC pages directly.
        checksums: list[str] | None = None
        if failure or not verify_checksum:
            checksums = None
        elif client_tensors is not None and num_full_tokens > 0:
            if pass_label == "cold":
                checksums = cold_ground_truth
            elif pass_label == "warm" and hit_chunks > 0:
                retr_num_blocks = hit_tokens // block_size
                checksums = _compute_client_checksums(
                    client_tensors,
                    block_offset,
                    retr_num_blocks,
                    block_size,
                    chunk_size,
                )
        elif http_base and num_full_tokens > 0:
            checksums = _query_checksum(
                http_base,
                block_offset,
                num_blocks,
                block_size,
                chunk_size,
                instance_id=instance_id,
            )
        if checksums:
            digest = hashlib.md5("".join(checksums).encode()).hexdigest()[:16]
            emit(
                "  [w%d seq %d/%s] CHECKSUM: %s (%d chunks)"
                % (
                    worker_id,
                    seq_no,
                    pass_label,
                    digest,
                    len(checksums),
                )
            )

        result = RequestResult(
            checksums=checksums,
            lookup_ms=lookup_ms,
            retrieve_ms=retrieve_ms if hit_chunks > 0 else None,
            store_ms=store_ms if miss_chunks > 0 else None,
            hit_chunks=hit_chunks,
            total_chunks=total_chunks,
            store_tokens=store_tokens,
            retrieve_tokens=retrieve_tokens,
            store_status=store_status,
            retrieve_status=retrieve_status,
            failure=failure,
        )
        return result
    finally:
        # The LOOKUP was accepted, so the server created session state;
        # release it on every exit path (normal return, an early failure
        # return, or an unexpected exception). END_SESSION only clears the
        # session, though -- it does NOT prove the prefetch job, read locks,
        # or a pending transfer left by an in-flight request are gone.
        #
        # END_SESSION is itself a submit-then-unknown RPC: _call submits it
        # before blocking on the reply and only converts a TimeoutError into
        # the _TIMEOUT sentinel, so any other error (ZMQ / server exception)
        # or a Ctrl-C while waiting propagates. If that escaped uncaught the
        # taint verdict would be lost (and a cleanup error could even mask the
        # body's), so guard it and turn a raised cleanup into an explicit
        # taint, chaining the cleanup exception.
        try:
            end_ok = _send_end_session(client, request_id)
        except KeyboardInterrupt as exc:
            raise ServerTaintedInterrupt(
                "interrupted while ending session (seq %d, %s pass); server "
                "effect unknown, restart required" % (seq_no, pass_label)
            ) from exc
        except Exception as exc:
            raise ServerTaintedError(
                "END_SESSION failed with unknown server state (seq %d, %s "
                "pass); restart required" % (seq_no, pass_label)
            ) from exc
        if result is not None:
            # A result was produced: the request ran its whole lifecycle
            # (prefetch consumed, contract paths freed their locks) or is an
            # already-tainted early failure. Only a failed END_SESSION adds
            # taint here; a successful one leaves the result's verdict intact.
            if not end_ok:
                # Mark the result tainted AND turn it into a failure so the
                # driver aborts the worker immediately (fail-close) rather
                # than continuing to issue requests against a tainted server.
                result.server_tainted = True
                result.failure = result.failure or (
                    "END_SESSION timeout (seq %d, %s pass); server tainted, "
                    "restart required" % (seq_no, pass_label)
                )
        else:
            # The body raised (an unexpected gather/shape/checksum error, or
            # a Ctrl-C) before producing a result, so the request was cut off
            # in flight. Its server-side effects are indeterminate: the
            # prefetch job is removed only once QUERY_PREFETCH_STATUS reads
            # completion, and an interrupted transfer may leave read locks or
            # a pending transfer -- none of which a successful END_SESSION
            # clears. So taint the server whether or not END_SESSION acked,
            # chaining the original exception so it is not lost.
            body_exc = sys.exc_info()[1]
            if not end_ok:
                reason = (
                    "%s and END_SESSION timed out (seq %d, %s pass); server "
                    "restart required"
                )
            else:
                reason = (
                    "%s mid-request; END_SESSION acked but prefetch job / read "
                    "locks / pending transfer may remain (seq %d, %s pass); "
                    "server restart required"
                )
            if isinstance(body_exc, KeyboardInterrupt):
                # Keep interrupt semantics (the run still exits 130) but flag
                # the taint so the server is reported unsafe to reuse.
                raise ServerTaintedInterrupt(
                    reason % ("interrupted", seq_no, pass_label)
                ) from body_exc
            raise ServerTaintedError(
                reason % ("request failed", seq_no, pass_label)
            ) from body_exc


# ------------------------------------------------------------------ #
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(client: MessageQueueClient) -> int:
    """Query the server's chunk size."""
    result = _call(client, RequestType.GET_CHUNK_SIZE, [])
    if result is _TIMEOUT or result is None:
        return 256  # fallback
    return int(result)
