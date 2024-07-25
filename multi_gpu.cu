#include <stdio.h>

#define MIN_GLOBAL_MEMORY 4LL * 1024 * 1024 * 1024 // 4 GB in bytes

// CUDA kernel function (simple example)
__global__ void simpleKernel() {
    printf("Hello from the GPU!\n");
}

int main() {
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    printf("Number of GPU Devices: %d\n", nDevices);

    int currentChosenDeviceNumber = -1;

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device Compute Major: %d Minor: %d\n", prop.major, prop.minor);
        printf("  Max Thread Dimensions: [%d][%d][%d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Number of Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Device Clock Rate (KHz): %d\n", prop.clockRate);
        printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
        printf("  Registers Per Block: %d\n", prop.regsPerBlock);
        printf("  Registers Per Multiprocessor: %d\n", prop.regsPerMultiprocessor);
        printf("  Shared Memory Per Block: %zu\n", prop.sharedMemPerBlock);
        printf("  Shared Memory Per Multiprocessor: %zu\n", prop.sharedMemPerMultiprocessor);
        printf("  Total Constant Memory (bytes): %zu\n", prop.totalConstMem);
        printf("  Total Global Memory (bytes): %zu\n", prop.totalGlobalMem);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

        // Check if the device meets the minimum global memory requirement
        if (prop.totalGlobalMem >= MIN_GLOBAL_MEMORY) {
            currentChosenDeviceNumber = i;
            break;
        }
    }

    // Print out the chosen device
    printf("The chosen GPU device has an index of: %d\n", currentChosenDeviceNumber);

    if (currentChosenDeviceNumber != -1) {
        // Set the chosen device
        cudaSetDevice(currentChosenDeviceNumber);

        // Execute a simple kernel on the chosen device
        simpleKernel<<<1, 1>>>();
        cudaDeviceSynchronize();  // Ensure the kernel execution is completed
    } else {
        printf("No suitable GPU device found with the required minimum global memory.\n");
    }

    return 0;
}
