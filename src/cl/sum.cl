#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_global(__global unsigned int* dest, __global const unsigned int *array, unsigned int n) {
    unsigned int id = get_global_id(0);

    if (id >= n) return;

    atomic_add(dest, array[id]);
}


#define SUM_LOOP_ENTRIES_PER_THREAD 32
__kernel void sum_loop(__global unsigned int* dest, __global const unsigned int *array, unsigned int n) {
    unsigned int id = get_global_id(0);

    unsigned int sum = 0;
    for (unsigned int i = 0; i < SUM_LOOP_ENTRIES_PER_THREAD; i++) {
        sum += array[id * SUM_LOOP_ENTRIES_PER_THREAD + i];
    }

    atomic_add(dest, sum);
}

#define SUM_LOOP_C_ENTRIES_PER_THREAD 32
__kernel void sum_loop_coalesced(__global unsigned int* dest, __global const unsigned int *array, unsigned int n) {
    unsigned int group_size = get_local_size(0);
    unsigned int group_id = get_group_id(0);
    unsigned int id = get_local_id(0);

    unsigned int entries_per_group = SUM_LOOP_C_ENTRIES_PER_THREAD * group_size;

    unsigned int sum = 0;
    for (unsigned int i = 0; i < SUM_LOOP_ENTRIES_PER_THREAD; i++) {
        sum += array[group_id * entries_per_group + i * group_size + id];
    }

    atomic_add(dest, sum);
}


#define WORK_GROUP_SIZE 256
__kernel void sum_local(__global unsigned int* dest, __global const unsigned int *array, unsigned int n) {
    unsigned int group_size = get_local_size(0);
    unsigned int local_id = get_local_id(0);
    unsigned int global_id = get_global_id(0);

    __local unsigned int local_data[WORK_GROUP_SIZE];
    local_data[local_id] = array[global_id];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (local_id == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < WORK_GROUP_SIZE; i++) {
            sum += local_data[i];
        }

        atomic_add(dest, sum);
    }
}

