"""GPU kernels for the cross-sectional half of the panel API.

Which half is not arbitrary. The cross-sectional ops are `O(T * P^2)` for rank
(every entity compared against every other, each bar) and `O(T * P)` for
z-score/demean, with **no dependence between rows** — thousands of independent
small reductions is the shape a GPU is built for, and the arithmetic intensity of
rank in particular is high enough to pay for the transfer.

The rolling ops are the opposite: a running-sum recurrence along time, `O(T * P)`
with a serial dependence per entity. There is a parallel-scan formulation, but at
realistic panel sizes it loses to the CPU's cache-resident single pass plus the
PCIe round trip, so it is deliberately absent rather than present and slower.

One block per bar, one thread per entity: the row is staged in shared memory, so
the `P^2` comparisons of a rank read from shared rather than global, and z-score
does a tree reduction instead of an atomic storm.
"""

from max.gpu import barrier, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.math import isnan, nan, sqrt
from std.memory import stack_allocation

comptime NAN = nan[DType.float64]()
comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]

comptime BLOCK = 256
# Entities per bar that fit in the shared row buffer: 2048 doubles = 16 KiB,
# inside the 48 KiB default and wider than any real cross-section.
comptime MAX_COLS = 2048

comptime ERR_UNAVAILABLE = -1
comptime ERR_ARGS = -2
comptime ERR_TOO_WIDE = -3


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def xs_rank_kernel(
    x: UnsafePointer[Float64, AnyOrigin[mut=True]],
    valid: UnsafePointer[Int64, AnyOrigin[mut=True]],
    dst: UnsafePointer[Float64, AnyOrigin[mut=True]],
    n_rows: Int32,
    n_cols: Int32,
    signed: Int32,
):
    var t = Int32(block_idx.x)
    if t >= n_rows:
        return
    var row = stack_allocation[MAX_COLS, Float64, address_space = AddressSpace.SHARED]()
    var base = t * n_cols
    var tx = Int32(thread_idx.x)

    # Stage the row once; every comparison below then reads shared memory.
    var j = tx
    while j < n_cols:
        row[j] = NAN if valid[base + j] == 0 else x[base + j]
        j += BLOCK
    barrier()

    var count = Int32(0)
    var k = Int32(0)
    while k < n_cols:
        if not isnan(row[k]):
            count += 1
        k += 1

    j = tx
    while j < n_cols:
        if isnan(row[j]) or count < 2:
            dst[base + j] = NAN
        else:
            var v = row[j]
            var less = Int32(0)
            var k = Int32(0)
            while k < n_cols:
                var u = row[k]
                if not isnan(u) and (u < v or (u == v and k < j)):
                    less += 1
                k += 1
            var pct = Float64(less) / Float64(count - 1)
            dst[base + j] = pct * 2.0 - 1.0 if signed != 0 else pct
        j += BLOCK


def xs_zscore_kernel(
    x: UnsafePointer[Float64, AnyOrigin[mut=True]],
    valid: UnsafePointer[Int64, AnyOrigin[mut=True]],
    dst: UnsafePointer[Float64, AnyOrigin[mut=True]],
    n_rows: Int32,
    n_cols: Int32,
    clip: Float64,
):
    var t = Int32(block_idx.x)
    if t >= n_rows:
        return
    var ssum = stack_allocation[BLOCK, Float64, address_space = AddressSpace.SHARED]()
    var ssq = stack_allocation[BLOCK, Float64, address_space = AddressSpace.SHARED]()
    var scnt = stack_allocation[BLOCK, Float64, address_space = AddressSpace.SHARED]()
    var base = t * n_cols
    var tx = Int32(thread_idx.x)

    var total = 0.0
    var sq = 0.0
    var cnt = 0.0
    var j = tx
    while j < n_cols:
        if valid[base + j] != 0:
            var v = x[base + j]
            if not isnan(v):
                total += v
                sq += v * v
                cnt += 1.0
        j += BLOCK
    ssum[tx] = total
    ssq[tx] = sq
    scnt[tx] = cnt
    barrier()

    var stride = Int32(BLOCK // 2)
    while stride > 0:
        if tx < stride:
            ssum[tx] += ssum[tx + stride]
            ssq[tx] += ssq[tx + stride]
            scnt[tx] += scnt[tx + stride]
        barrier()
        stride //= 2

    var n = scnt[0]
    var mean = 0.0
    var var_ = 0.0
    if n > 0.0:
        mean = ssum[0] / n
        var_ = ssq[0] / n - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    var sd = sqrt(var_)
    if sd <= 1e-12:
        sd = 1.0

    j = tx
    while j < n_cols:
        if valid[base + j] == 0 or isnan(x[base + j]) or n < 2.0:
            dst[base + j] = NAN
        else:
            var z = (x[base + j] - mean) / sd
            if clip > 0.0:
                if z > clip:
                    z = clip
                elif z < -clip:
                    z = -clip
            dst[base + j] = z
        j += BLOCK


@export("mp_gpu_available")
def mp_gpu_available() abi("C") -> Int:
    """1 when a device context can be created right now, else 0 (never raises)."""
    try:
        var ctx = DeviceContext()
        _ = ctx.name()
        return 1
    except:
        return 0


@export("mp_gpu_xs_rank")
def mp_gpu_xs_rank(
    x_addr: Int, valid_addr: Int, out_addr: Int, n_rows: Int, n_cols: Int, signed: Int
) abi("C") -> Int:
    if n_rows <= 0 or n_cols <= 0 or x_addr == 0 or valid_addr == 0 or out_addr == 0:
        return ERR_ARGS
    if n_cols > MAX_COLS:
        return ERR_TOO_WIDE
    var total = n_rows * n_cols
    try:
        var ctx = DeviceContext()
        var dx = ctx.enqueue_create_buffer[DType.float64](total)
        var dv = ctx.enqueue_create_buffer[DType.int64](total)
        var dout = ctx.enqueue_create_buffer[DType.float64](total)
        ctx.enqueue_copy(dx, fp(x_addr))
        ctx.enqueue_copy(dv, ip(valid_addr))
        ctx.enqueue_function[xs_rank_kernel](
            dx, dv, dout, Int32(n_rows), Int32(n_cols), Int32(signed),
            grid_dim=n_rows, block_dim=BLOCK,
        )
        ctx.enqueue_copy(fp(out_addr), dout)
        ctx.synchronize()
        return 0
    except:
        return ERR_UNAVAILABLE


@export("mp_gpu_xs_zscore")
def mp_gpu_xs_zscore(
    x_addr: Int, valid_addr: Int, out_addr: Int, n_rows: Int, n_cols: Int, clip: Float64
) abi("C") -> Int:
    if n_rows <= 0 or n_cols <= 0 or x_addr == 0 or valid_addr == 0 or out_addr == 0:
        return ERR_ARGS
    var total = n_rows * n_cols
    try:
        var ctx = DeviceContext()
        var dx = ctx.enqueue_create_buffer[DType.float64](total)
        var dv = ctx.enqueue_create_buffer[DType.int64](total)
        var dout = ctx.enqueue_create_buffer[DType.float64](total)
        ctx.enqueue_copy(dx, fp(x_addr))
        ctx.enqueue_copy(dv, ip(valid_addr))
        ctx.enqueue_function[xs_zscore_kernel](
            dx, dv, dout, Int32(n_rows), Int32(n_cols), clip,
            grid_dim=n_rows, block_dim=BLOCK,
        )
        ctx.enqueue_copy(fp(out_addr), dout)
        ctx.synchronize()
        return 0
    except:
        return ERR_UNAVAILABLE
