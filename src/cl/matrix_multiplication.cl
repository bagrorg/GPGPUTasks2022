#define TILE_SIZE 16

/**
    A \in M_(R, rows: M, cols: K)
    B \in M_(R, K, N)
    C \in M_(R, M, N)
*/
__kernel void matrix_multiplication(__global const float *A, __global const float *B, __global float *C, unsigned int M, unsigned int N, unsigned int K) {
    __local float tile_A[TILE_SIZE * TILE_SIZE];
    __local float tile_B[TILE_SIZE * TILE_SIZE];

    unsigned int l_x = get_local_id(0);
    unsigned int l_y = get_local_id(1);

    unsigned int x = get_global_id(0);
    unsigned int y = get_global_id(1);

    double sum = 0;

    for (unsigned int i = 0; i < K; i += TILE_SIZE) {
        unsigned int index_x_A = l_x + i;
        unsigned int index_x_B = x;
        unsigned int index_y_A = y;
        unsigned int index_y_B = l_y + i;

        tile_A[l_x + l_y * TILE_SIZE] = A[index_y_A * K + index_x_A];
        tile_B[l_x + l_y * TILE_SIZE] = B[index_y_B * N + index_x_B];

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int kx = get_local_id(0);
        unsigned int ky = get_local_id(1);
        for (unsigned int i = 0; i < TILE_SIZE; i++) {
            sum += tile_A[i + ky * TILE_SIZE] * tile_B[kx + i * TILE_SIZE];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    C[y * N + x] = sum;
}
