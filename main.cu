#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>

#include <GLFW/glfw3.h>

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

static const int WIN_WIDTH = 500;                 // ウィンドウの幅
static const int WIN_HEIGHT = 500;                 // ウィンドウの高さ
static const std::string WIN_TITLE = "OpenGL Course";     // ウィンドウのタイトル

// ユーザ定義のOpenGLの初期化
void initializeGL()
{
    // 背景色の設定
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
}

// ユーザ定義のOpenGL描画
void paintGL() 
{
    // 背景色の描画
    glClear(GL_COLOR_BUFFER_BIT);
}

int main() 
{
    // OpenGLを初期化する
    if (glfwInit() == GL_FALSE) {
        fprintf(stderr, "Initialization failed!\n");
        return 1;
    }

    // Windowの作成
    GLFWwindow* window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE.c_str(),NULL, NULL);
    if (window == NULL) {
        fprintf(stderr, "Window creation failed!");
        glfwTerminate();
        return 1;
    }

    // OpenGLの描画対象にWindowを追加
    glfwMakeContextCurrent(window);

    // 初期化
    initializeGL();

    // メインループ
    while (glfwWindowShouldClose(window) == GL_FALSE) 
    {
        // 描画
        paintGL();

        // 描画用バッファの切り替え
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return 0;
}