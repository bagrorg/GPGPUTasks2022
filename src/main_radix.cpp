#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <climits>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)



void doPrefix(ocl::Kernel &prefix_step, ocl::Kernel &reduce_step, uint workGroupSize, uint global_work_size, uint n,
              gpu::gpu_mem_32u &as_gpu, gpu::gpu_mem_32u &bs_gpu, gpu::gpu_mem_32u &as_buffer_gpu) {
    for (uint b_st = 0; (1 << b_st) <= n; b_st++) {
        prefix_step.exec(gpu::WorkSize(workGroupSize, global_work_size),
                as_gpu, bs_gpu, n, b_st);

        reduce_step.exec(gpu::WorkSize(workGroupSize, global_work_size / 2),
                as_gpu, as_buffer_gpu, n >> (b_st + 1));
        
        std::swap(as_gpu, as_buffer_gpu);
    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (((unsigned int) r.next(0, std::numeric_limits<int>::max())));
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    
    unsigned int workGroupSize = 128;
    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    unsigned int workGroupCount = global_work_size / workGroupSize;

    gpu::gpu_mem_32u as_gpu, bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    uint32_t cnt_of_batches = 1 << 4;
    uint32_t hists_size = cnt_of_batches * workGroupCount;
    gpu::gpu_mem_32u hist_gpu;
    hist_gpu.resizeN(hists_size);
    uint hist_kernel_work = (hists_size + workGroupSize - 1) / workGroupSize * workGroupSize;

    // Prefix sum data
    gpu::gpu_mem_32u p_bs_gpu, p_buffer;
    p_bs_gpu.resizeN(hists_size);
    p_buffer.resizeN(hists_size);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        ocl::Kernel count_step(radix_kernel, radix_kernel_length, "count_step");
        ocl::Kernel prefix_step(radix_kernel, radix_kernel_length, "prefix_step");
        ocl::Kernel prefix_reduce_step(radix_kernel, radix_kernel_length, "reduce_step");
        ocl::Kernel cleanup(radix_kernel, radix_kernel_length, "cleanup");
        radix.compile();
        count_step.compile();
        prefix_step.compile();
        prefix_reduce_step.compile();
        cleanup.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (uint b_st = 0; b_st < sizeof(uint) * CHAR_BIT; b_st += 4) {
                count_step.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, hist_gpu, b_st, workGroupCount);

                cleanup.exec(gpu::WorkSize(workGroupSize, hist_kernel_work), p_bs_gpu);        
                doPrefix(prefix_step, prefix_reduce_step, workGroupSize, hist_kernel_work, hists_size, hist_gpu, p_bs_gpu, p_buffer);

                radix.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, p_bs_gpu, b_st, workGroupCount);

                std::swap(as_gpu, bs_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    
    

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}