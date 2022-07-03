#include <iostream>
#include <stdio.h>
#include <array>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <sstream>

#include <cuda.h>
#include <cuda_runtime.h>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdlib>

#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include "shader.hpp"

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

// ���_�z��I�u�W�F�N�g�̍쐬
static GLuint createObject(GLuint vertices, const GLfloat(*position)[2])
{
    // ���_�z��I�u�W�F�N�g
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // ���_�o�b�t�@�I�u�W�F�N�g
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 2 * vertices, position, GL_STATIC_DRAW);

    // ��������Ă��钸�_�o�b�t�@�I�u�W�F�N�g�� attribute �ϐ�����Q�Ƃł���悤�ɂ���
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    // ���_�o�b�t�@�I�u�W�F�N�g�ƒ��_�z��I�u�W�F�N�g�̌�������������
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    return vao;
}

static const int WIN_WIDTH = 500;                 // �E�B���h�E�̕�
static const int WIN_HEIGHT = 500;                 // �E�B���h�E�̍���
static const std::string WIN_TITLE = "OpenGL Course";     // �E�B���h�E�̃^�C�g��

int main() 
{
    // OpenGL������������
    if (glfwInit() == GL_FALSE) {
        std::cerr << "Can't initialize GLFW" << std::endl;
        return 1;
    }

    atexit(glfwTerminate);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Window�̍쐬
    GLFWwindow* window = glfwCreateWindow(WIN_WIDTH, WIN_HEIGHT, WIN_TITLE.c_str(),NULL, NULL);
    if (window == NULL) {
        std::cerr << "Can't create GLFW window." << std::endl;
        return 1;
    }

    // OpenGL�̕`��Ώۂ�Window��ǉ�
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        // GLEW �̏������Ɏ��s����
        std::cerr << "Can't initialize GLEW" << std::endl;
        return 1;
    }

    glfwSwapInterval(1);

    // �o�[�e�b�N�X�V�F�[�_�̃\�[�X�v���O����
    static const GLchar vsrc[] =
        "#version 150 core\n"
        "in vec4 pv;\n"
        "void main(void)\n"
        "{\n"
        "  gl_Position = pv;\n"
        "}\n";

    // �t���O�����g�V�F�[�_�̃\�[�X�v���O����
    static const GLchar fsrc[] =
        "#version 150 core\n"
        "out vec4 fc;\n"
        "void main(void)\n"
        "{\n"
        "  fc = vec4(1.0, 0.0, 0.0, 0.0);\n"
        "}\n";

    // ������
    GLuint program = createProgram(vsrc, "pv", fsrc, "fc");
    // �}�`�f�[�^
    static const GLfloat position[][2] =
    {
      { -0.5f, -0.5f },
      {  0.5f, -0.5f },
      {  0.5f,  0.5f },
      { -0.5f,  0.5f }
    };
    static const int vertices = sizeof position / sizeof position[0];

    // ���_�z��I�u�W�F�N�g�̍쐬
    GLuint vao = createObject(vertices, position);

    // ���C�����[�v
    while (glfwWindowShouldClose(window) == GL_FALSE) 
    {
        glClear(GL_COLOR_BUFFER_BIT);
        // �`��
        glUseProgram(program);

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_LOOP, 0, vertices);
        glBindVertexArray(0);

        // �`��p�o�b�t�@�̐؂�ւ�
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return 0;
}