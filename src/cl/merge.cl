/**
        .....  [e1, e2, e3, e4....., em | q1, q2, q3, q4, ......., qm]  ........
               ^                          ^                          ^
               |                          |                          |
              start                      mid

               <----------------------------------------------------->
                                        level
 */
__kernel void merge(__global const float *array, unsigned int n, unsigned int level, __global float *out) {
    unsigned int x = get_global_id(0);
    if (x >= n) return;
    
    unsigned int part_size = level / 2;
    unsigned int start     = x / level * level;
    unsigned int mid       = start + part_size;
    unsigned int local_id  = x % part_size;
    
    float element = array[x];
    unsigned int id = 0;

    {
        size_t l = 0;
        size_t r = part_size;
        __global const float *second_array;

        if (x < mid) {
            second_array = array + mid;
        } else {
            second_array = array + start;
        }


        while (l < r - 1) {
            size_t m = (l + r) / 2;
            float second_array_element = second_array[m];
            if ((second_array_element > element && x < mid) || 
                (second_array_element >= element && x >= mid)) {
                r = m;
                continue;
            } else {
                l = m;
                continue;
            }
        }

        float second_array_element = second_array[l];
        if ((second_array_element > element && x < mid) || 
            (second_array_element >= element && x >= mid)) id = local_id;
        else id = local_id + r;
    }

    out[id + start] = element;
}



