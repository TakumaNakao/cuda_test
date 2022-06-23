#include <stdio.h>
#include <array>
#include <vector>
#include <chrono>

#include <cuda.h>
#include<cuda_runtime.h>

#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace cuda_function
{
    template<typename T>
    __global__ void vector_add(T* a,T* b,T* c,size_t n)
    {
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

template<typename T>
class TestClass {
public:
    TestClass()
    {

    }
    TestClass(std::vector<T> v) : v_(v)
    {

    }
    std::vector<T> add(std::vector<T> v)
    {
        assert(v_.size() == v.size());
        auto start_time = std::chrono::system_clock::now();
        T* d_a;
        T* d_b;
        T* d_c;
        size_t bytes = v_.size() * sizeof(T);
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        cudaMemcpy(d_a, v_.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, v.data(), bytes, cudaMemcpyHostToDevice);
        size_t block_size = 32;
        dim3 dim_block(block_size);
        dim3 dim_grid(ceil(v_.size() / (float)block_size));
        cuda_function::vector_add <<<dim_grid, dim_block >>> (d_a, d_b, d_c, v_.size());
        cudaDeviceSynchronize();
        T* h_c = (T*)malloc(bytes);
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
        std::vector<T> ret;
        ret.reserve(v_.size());
        for (size_t i = 0; i < v_.size(); i++) {
            ret.push_back(h_c[i]);
        }
        auto end_time = std::chrono::system_clock::now();
        float elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        printf("elapsed time = %f[ms]\n", elapsed_time);
        return ret;
    }
private:
    std::vector<T> v_;
};


int main() {
    std::vector<float> vec_a;
    std::vector<float> vec_b;
    for (size_t i = 0; i < 1000000; i++) {
        vec_a.push_back(i);
        vec_b.push_back(i);
    }
    TestClass<float> a(vec_a);
    auto c = a.add(vec_b);
    return 0;
}