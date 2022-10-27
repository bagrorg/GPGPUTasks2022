#define LOCAL_SIZE 256

void download_to_local(__global float *as, __local float *l, uint pos, int arrow_step) {
    l[(pos) % (LOCAL_SIZE * 2)] = as[pos];
    l[(pos + arrow_step) % (LOCAL_SIZE * 2)] = as[pos + arrow_step];
}

void upload_from_local(__global float *as, __local float *l, uint pos, int arrow_step) {
    as[pos] = l[(pos) % (LOCAL_SIZE * 2)];
    as[pos + arrow_step] = l[(pos + arrow_step) % (LOCAL_SIZE * 2)];
}

void get_positions(uint th_id, uint block_number, uint local_block_number, uint *pos, int *arrow_step) {
    uint local_diff = (block_number - local_block_number);
    *arrow_step = 1 << local_diff;

    uint block_size = (1 << (block_number + 1));
    uint up = ((th_id % block_size) >= (block_size / 2));

    uint local_block_size = *arrow_step * 2;

    if (up) {
        *pos = (th_id / (local_block_size / 2) + 1) * local_block_size - 1 - (local_block_size / 2 - (th_id) % (local_block_size / 2) - 1);
        *arrow_step *= -1;
    } else {
        *pos = (th_id / (local_block_size / 2)) * local_block_size + (th_id % (local_block_size / 2));
    }
}

__kernel void local_bitonic(__global float *as, uint block_number, uint local_block_number, uint n) {
    int th_id = get_global_id(0);
    __local float l[LOCAL_SIZE * 2];
    uint size = LOCAL_SIZE * 2;
    //if (th_id >= n / 2) return;

    uint pos;
    int arrow_step;
    get_positions(th_id, block_number, local_block_number, &pos, &arrow_step);
    download_to_local(as, l, pos, arrow_step);
    
    while (local_block_number <= block_number) {
        float fst = l[pos % size];
        float snd = l[(pos + arrow_step) % size];

        if (fst > snd) {
            float tmp = fst;
            fst = snd;
            snd = tmp;
        } 

        l[pos % size] = fst;
        l[(pos + arrow_step) % size] = snd;

        local_block_number += 1;
        get_positions(th_id, block_number, local_block_number, &pos, &arrow_step);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    get_positions(th_id, block_number, local_block_number - 1, &pos, &arrow_step);
    upload_from_local(as, l, pos, arrow_step);
}


//  block_number \in 0..
//  local_block_number \in 0..
__kernel void bitonic(__global float *as, uint block_number, uint local_block_number, uint n) {
    // TODO
    int th_id = get_global_id(0);
    if (th_id >= n / 2) return;
    
    uint pos;
    int arrow_step;
    get_positions(th_id, block_number, local_block_number, &pos, &arrow_step);
    

    float fst = as[pos];
    float snd = as[pos + arrow_step];

    if (fst > snd) {
        float tmp = fst;
        fst = snd;
        snd = tmp;
    } 

    as[pos] = fst;
    as[pos + arrow_step] = snd;
}
