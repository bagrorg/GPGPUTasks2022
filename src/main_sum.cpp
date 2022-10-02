#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/shared_device_buffer.h>
#include <libgpu/context.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void benchGpuAlgo(const std::string kernelName, const std::vector<unsigned int> &as, size_t benchmarkingIters, unsigned int reference_sum, size_t work_group_size, size_t global_work_size) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
    
    bool printLog = false;
    kernel.compile(printLog);

    gpu::gpu_mem_32f array, dest;
    unsigned int ans = 0;
    
    array.resizeN(as.size());
    array.write(as.data(), as.size() * sizeof(unsigned int));

    dest.resizeN(1);

    timer t;
    for (size_t i = 0; i < benchmarkingIters; i++) {
        ans = 0;
        dest.write(&ans, sizeof(unsigned int));
        kernel.exec(gpu::WorkSize(work_group_size, global_work_size),
                    dest, array, (unsigned int) as.size());
        dest.read(&ans, sizeof(unsigned int));
        EXPECT_THE_SAME(reference_sum, ans, "GPU result should be consistent!");
        t.nextLap();
    }
    std::cout << "GPU " + kernelName + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " + kernelName + ": " << (as.size()/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    std::cout << std::endl;

}


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }
    std::cout << "\n\n ==============================  CPU RESULTS ============================== \n\n";
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    std::cout << "\n\n ==============================  OpenCL RESULTS ============================== \n\n";

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    size_t bufSize = (as.size() + 256 - 1) / 256 * 256;
    as.resize(bufSize, 0);


    {
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;    // always as.size()

        benchGpuAlgo("sum_global", as, benchmarkingIters, reference_sum, workGroupSize, global_work_size);
    }
    {
        size_t count_of_elements_per_thread = 64;
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
        global_work_size /= count_of_elements_per_thread;

        benchGpuAlgo("sum_loop", as, benchmarkingIters, reference_sum, workGroupSize, global_work_size);
    }
    {
        size_t count_of_elements_per_thread = 64;
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
        global_work_size /= count_of_elements_per_thread;
        
        benchGpuAlgo("sum_loop_coalesced", as, benchmarkingIters, reference_sum, workGroupSize, global_work_size);
    }
    {   
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;

        benchGpuAlgo("sum_local", as, benchmarkingIters, reference_sum, workGroupSize, global_work_size);
    }
    {   
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;

        benchGpuAlgo("sum_local_tree", as, benchmarkingIters, reference_sum, workGroupSize, global_work_size);
    }
}
