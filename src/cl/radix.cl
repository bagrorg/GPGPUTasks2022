#define BIT_BATCH_SIZE 4
#define TWO_POW_K (1 << BIT_BATCH_SIZE)
#define MASK 15                         // 0b00....01111

// MUST BE >= TWO_POW_K
#define WG_SIZE 128

#define GET_BIT(num, i) (num >> i) & 1

uint get_batch(uint val, uint bit) {
    return (val >> bit) & MASK;
}


//////////////////////////////////////////////////////////////////////////////////////////
//
//                                      COUNT STEP
//
//////////////////////////////////////////////////////////////////////////////////////////


__kernel void count_step(__global const uint *as, __global uint *hist, uint bit, uint WG_CNT) {
    uint th_id = get_global_id(0);
    uint lth_id = get_local_id(0);
    uint wg_id = get_group_id(0);

    __local uint lhists[TWO_POW_K * WG_SIZE];
    for (uint elem = 0; elem < TWO_POW_K; elem++) {
        lhists[lth_id * TWO_POW_K + elem] = 0;
    }


    uint addr = WG_SIZE * wg_id + lth_id;

    uint element = as[addr];
    uint element_batch = get_batch(element, bit);
    lhists[lth_id * TWO_POW_K + element_batch]++;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lth_id < TWO_POW_K) {
        uint sum = 0;
        for (uint i = 0; i < WG_SIZE; i++) {
            sum += lhists[i * TWO_POW_K + lth_id];
        }

        hist[lth_id * WG_CNT + wg_id] = sum;
    }
}


//////////////////////////////////////////////////////////////////////////////////////////
//
//                                      PREFIX STEP
//
//////////////////////////////////////////////////////////////////////////////////////////

__kernel void naive_prefix(__global uint *as, uint n) {
    if (get_global_id(0) == 0) {
        uint cnt = 0;
        for (uint i = 0; i < n; i++) {
            uint tmp = as[i];
            as[i] = cnt;
            cnt += tmp;
        }
    }
}

__kernel void prefix_step(__global uint *as, __global uint *bs, uint n, uint bit) {
    uint th_id = get_global_id(0);
    uint elem_id = th_id + 1;       // Because we need to iterate from `1`

    if (th_id >= n) return;

    if (GET_BIT(elem_id, bit)) {
        if (elem_id < n)
            bs[elem_id] += as[(elem_id >> bit) - 1];
    }
}

__kernel void reduce_step(__global uint *as, __global uint *new_as, uint n) {
    uint th_id = get_global_id(0);

    if (th_id >= n) return;

    new_as[th_id] = as[2 * th_id] + as[2 * th_id + 1];
}

__kernel void cleanup(__global uint *vec) {
    vec[get_global_id(0)] = 0;
}

//////////////////////////////////////////////////////////////////////////////////////////
//
//                                      REORDER STEP
//
//////////////////////////////////////////////////////////////////////////////////////////


void local_count(__local uint *ldata, uint bit, __local uint *hists) {
    uint lth_id = get_local_id(0);
    for (uint elem = 0; elem < TWO_POW_K; elem++) {
        hists[elem * WG_SIZE + lth_id] = 0;
    }

    uint element = ldata[lth_id];
    uint element_batch = get_batch(element, bit);
    hists[element_batch * WG_SIZE + lth_id]++;
}

void local_prefix(__local uint *data, uint n) {
    uint count = 0;
    for (int i = 0; i < n; i++) {
        uint tmp = data[i];
        data[i] = count;
        count += tmp;
    }
}

__kernel void radix(__global unsigned int *as, __global uint *out, __global uint *hist, uint bit, uint WG_CNT) {
    uint th_id  = get_global_id(0);
    uint lth_id = get_local_id(0);
    uint wg_id  = get_group_id(0);

    __local uint ldata[WG_SIZE];
    __local uint local_hists[WG_SIZE * TWO_POW_K];

    ldata[lth_id] = as[WG_SIZE * wg_id + lth_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    local_count(ldata, bit, local_hists);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lth_id < TWO_POW_K) {
        local_prefix(local_hists + lth_id * WG_SIZE, WG_SIZE);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint element = ldata[lth_id];
    uint element_batch = (element >> bit) & MASK;

    uint glob_id = hist[element_batch * WG_CNT + wg_id];
    uint loc_id = local_hists[element_batch * WG_SIZE + lth_id];

    uint id = glob_id + loc_id;

    out[id] = ldata[lth_id];
}
