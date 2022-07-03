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

    // ������
    GLint shader = makeShader("shader.vert", "shader.frag");
    // 2���̎O�p�|���S��
    std::vector<glm::vec3> positionList = {
        glm::vec3(0, 0, 1),glm::vec3(1,0, 0),glm::vec3(0, 0, 0),
        glm::vec3(0, 0, 1),glm::vec3(0, 0, 0),glm::vec3(0, 1, 0),
    };
    // attribute ���w�肷��
    GLint positionLocation = glGetAttribLocation(shader, "position");
    // ���_�o�b�t�@�I�u�W�F�N�g���쐬
    GLuint positionBuffer;
    glGenBuffers(1, &positionBuffer);
    // GPU���ɒ��_�o�b�t�@�I�u�W�F�N�g�Ƀ������̈���m�ۂ���
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * positionList.size(), positionList.data(), GL_STATIC_DRAW);

    GLuint matrixID = glGetUniformLocation(shader, "MVP");

    // ���C�����[�v
    while (glfwWindowShouldClose(window) == GL_FALSE) 
    {
        // �`��
        glUseProgram(shader);

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClearColor(0.2f, 0.2f, 0.2f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // �錾���ɂ͒P�ʍs�񂪓����Ă���
        glm::mat4 modelMat, viewMat, projectionMat;

        // View�s����v�Z
        viewMat = glm::lookAt(
            glm::vec3(2.0, 2.0, 2.0), // ���[���h��Ԃł̃J�����̍��W
            glm::vec3(0.0, 0.0, 0.0), // ���Ă���ʒu�̍��W
            glm::vec3(0.0, 0.0, 1.0)  // ������������B(0,1.0,0)�ɐݒ肷���y������ɂȂ�܂�
        );

        // Projection�s����v�Z
        projectionMat = glm::perspective(
            glm::radians(45.0f), // �Y�[���̓x����(�ʏ�90�`30)
            (GLfloat)WIN_WIDTH / (GLfloat)WIN_HEIGHT,		// �A�X�y�N�g��
            0.1f,		// �߂��̃N���b�s���O����
            100.0f		// �����̃N���b�s���O����
        );

        // ModelViewProjection�s����v�Z
        glm::mat4 mvpMat = projectionMat * viewMat * modelMat;

        // ���݃o�C���h���Ă���V�F�[�_��uniform�ϐ�"MVP"�ɕϊ��s��𑗂�
        // 4�ڂ̈����͍s��̍ŏ��̃A�h���X��n���Ă��܂��B
        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvpMat[0][0]);

        // positionLocation�Ŏw�肳�ꂽattribute��L����
        glEnableVertexAttribArray(positionLocation);
        // positionBuffer�Ƀo�C���h
        glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
        // attribute�ϐ�position�Ɋ��蓖�Ă�
        // GPU���������ɑ����Ă������f�[�^���o�[�e�b�N�X�V�F�[�_�[�Ŏg���w��ł�
        glVertexAttribPointer(positionLocation, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

        glDrawArrays(GL_TRIANGLES, 0, 6);

        // �`��p�o�b�t�@�̐؂�ւ�
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    return 0;
}