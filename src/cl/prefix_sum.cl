#define GET_BIT(num, i) (num >> i) & 1

__kernel void prefix_step(__global uint *as, __global uint *bs, uint n, uint bit) {
    uint th_id = get_global_id(0);
    uint elem_id = th_id + 1;       // Because we need to iterate from `1`

    if (th_id >= n) return;

    if (GET_BIT(elem_id, bit)) {
        bs[th_id] += as[(elem_id >> bit) - 1];      // No out of range because elem_id >= 1
    }
}

__kernel void reduce_step(__global uint *as, __global uint *new_as, uint n) {
    uint th_id = get_global_id(0);

    if (th_id >= n) return;

    new_as[th_id] = as[2 * th_id] + as[2 * th_id + 1];
}

__kernel void cleanup(__global uint *vec, uint n) {
    if (get_global_id(0) < n)
        vec[get_global_id(0)] = 0;
}
