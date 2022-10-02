#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *gpu_image, unsigned int width, unsigned int height, 
                                  float fromX, float fromY, float sizeX, float sizeY, 
                                  unsigned int iterationsLimit, int smoothing, int aliasing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;

    for (; iter < iterationsLimit; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    
    float result = iter;
    
    if ((smoothing == 1) && (iter != iterationsLimit)) {
        result = result - native_log(native_log(sqrt(x * x + y * y)) / native_log(threshold)) / native_log(2.0f);
    }
    
    result = 1.0f * result / iterationsLimit;
    gpu_image[j * width + i] = result;
    
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (aliasing == 1) {
        int N = 3;
        
        result = 0;
        unsigned int cnt = 0;

        for (int ny = -1 * N / 2; ny <= N / 2; ny++) {
            for (int nx = -1 * N / 2; nx <= N / 2; nx++) {
                int jny = (int) j + ny;
                int inx = (int) i + nx;
                if ((jny >= 0 ) && (jny < height)) {
                    if ((inx >= 0 )&& (inx < width)) {
                        result += gpu_image[jny * width + inx];
                        cnt += 1;
                    }
                }
            }
        }

        result /= cnt;

        barrier(CLK_GLOBAL_MEM_FENCE);

        gpu_image[j * width + i] = result;
    }
}
