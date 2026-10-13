# 1. Publish this split's partial, writing through L2 to memory.
cdna4.buffer_store(
    ptr=partial_ptr + (split * num_tiles + tile) * tile_elems,
    offsets=partial_offsets,
    stored_value=acc,
    cache=".wt",
)
# 2. Wait until this wave's stores are acknowledged.
gl.inline_asm_elementwise(
    asm="s_waitcnt vmcnt(0)",
    constraints="=v,0",
    args=[partial_offsets],
    dtype=gl.int32,
    is_pure=False,
    pack=1,
)
# 3. Wait until every wave in the workgroup has done the same.
gl.barrier()
# 4. Announce arrival; only the counter needs to be atomic.
old = gl.atomic_add(counter_ptr + tile, 1, sem="relaxed", scope="gpu")
if old != SPLIT_K - 1:
    return  # Not last: another workgroup finishes the tile.
# 5. Last arrival: read all partials in split order, bypassing caches.
acc = cdna4.buffer_load(
    ptr=partial_ptr + tile * tile_elems, offsets=partial_offsets, cache=".cv"
)
for s in gl.static_range(1, SPLIT_K):
    acc += cdna4.buffer_load(
        ptr=partial_ptr + (s * num_tiles + tile) * tile_elems,
        offsets=partial_offsets,
        cache=".cv",
    )
# 6. Reset the counter for the next launch on this stream.
gl.store(counter_ptr + tile, 0)
# ... epilogue writes acc to the output ...

Proposition
the careful .wt -> vmcnt(0) -> barrier -> relaxed atomic -> .cv avoids full L2 buffer_wbl2 and buffer_inv which can be costly. yet manages correctness afaict

Comment  - 1
I have a suspicion that vmcnt for the memory store guarantees the data is arrived at HBM or at least visible to a consumer. Probably it might only tell us the store request/data has been acknowledged and we can discard the source. That's one of traditional difficulties of global synchronization within a kernel, not sure gfx950 has resolved it.

Commment - 2
We might not need the inline assembly because the gl.barrier() should lower to a atomic fence which needs to wait for vmcnt(0) anyways?

