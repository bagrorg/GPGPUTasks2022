#define TILE_SIZE 16

/**
    A \in M_(R, rows: M, cols: K)
    B \in M_(R, K, N)
    C \in M_(R, M, N)
*/
// TODO CHECK +1
__kernel void matrix_multiplication(__global const float *A, __global const float *B, __global float *C, unsigned int M, unsigned int N, unsigned int K) {
    __local float tile_A[(TILE_SIZE + 1) * TILE_SIZE];
    __local float tile_B[(TILE_SIZE + 1) * TILE_SIZE];

    unsigned int l_x = get_local_id(0);
    unsigned int l_y = get_local_id(1);

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    float sum = 0;

    for (unsigned int i = 0; i < K; i += TILE_SIZE) {
        unsigned int index_x_A = l_x + i;
        unsigned int index_x_B = x;
        unsigned int index_y_A = y;
        unsigned int index_y_B = l_y + i;

        tile_A[l_x + l_y * TILE_SIZE] = A[index_y_A * K + index_x_A];
        tile_B[l_x + l_y * TILE_SIZE] = B[index_y_B * N + index_x_B];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[i + l_y * TILE_SIZE] * tile_B[l_x + i * TILE_SIZE];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (y * N + x < M * N)
        C[y * N + x] = sum;
}

#define TILE_SIZE2 32
#define THREAD_WORK 8

__kernel void matrix_multiplication_fma(__global const float *A, __global const float *B, __global float *C, unsigned int M, unsigned int N, unsigned int K) {
    __local float tile_A[(TILE_SIZE2 + 1) * TILE_SIZE2];
    __local float tile_B[(TILE_SIZE2 + 1) * TILE_SIZE2];

    unsigned int l_x = get_local_id(0); // TILE_SIZE2 / THREAD_WORK
    unsigned int l_y = get_local_id(1);

    unsigned int x = get_global_id(0); // ..N / THREAD_WORK
    unsigned int y = get_global_id(1);

    float sum[THREAD_WORK];
    for (unsigned int i = 0; i < THREAD_WORK; i++) {
        sum[i] = 0;
    }

    for (unsigned int i = 0; i < K; i += TILE_SIZE2) {
        for (unsigned int w = 0; w < THREAD_WORK; w++) {
            unsigned int index_x_A = l_x * THREAD_WORK + w + i;
            unsigned int index_x_B = x * THREAD_WORK + w;
            unsigned int index_y_A = y;
            unsigned int index_y_B = l_y + i;

            tile_A[l_x * THREAD_WORK + w + l_y * TILE_SIZE2] = A[index_y_A * K + index_x_A];
            tile_B[l_x * THREAD_WORK + w + l_y * TILE_SIZE2] = B[index_y_B * N + index_x_B];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        
        for (unsigned int i = 0; i < TILE_SIZE2; i++) {
            for (unsigned int w = 0; w < THREAD_WORK; w++) {
                sum[w] += tile_A[i + l_y * TILE_SIZE2] * tile_B[l_x * THREAD_WORK + w + i * TILE_SIZE2];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    for (unsigned int w = 0; w < THREAD_WORK; w++) {
        if (y * N + x * THREAD_WORK + w < M * N)
            C[y * N + x * THREAD_WORK + w] = sum[w];
    }
}