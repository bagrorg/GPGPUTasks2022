#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *m, __global float *dest, unsigned int rows, unsigned int cols) {
	unsigned int id = get_global_id(1) * cols + get_global_id(0);
	unsigned int id_local = get_local_id(1) * TILE_SIZE + get_local_id(0);

	__local float tile[TILE_SIZE * TILE_SIZE];

	if (id >= rows * cols) tile[id_local] = 0;
	else { 
		float element = m[id];
		tile[id_local] = element;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	unsigned int x = get_group_id(1) * TILE_SIZE + get_local_id(0);
	unsigned int y = get_group_id(0) * TILE_SIZE + get_local_id(1);

	id_local = get_local_id(0) * TILE_SIZE + get_local_id(1);
	id = y * rows + x;

	if (id >= rows * cols) return;
	dest[y * rows + x] = tile[id_local];
}
