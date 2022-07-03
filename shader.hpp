#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdlib>

#define GLM_FORCE_CUDA
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

int readShaderSource(GLuint shaderObj, std::string fileName)
{
    //�t�@�C���̓ǂݍ���
    std::ifstream ifs(fileName);
    if (!ifs)
    {
        std::cout << "error" << std::endl;
        return -1;
    }

    std::string source;
    std::string line;
    while (getline(ifs, line))
    {
        source += line + "\n";
    }

    // �V�F�[�_�̃\�[�X�v���O�������V�F�[�_�I�u�W�F�N�g�֓ǂݍ���
    const GLchar* sourcePtr = (const GLchar*)source.c_str();
    GLint length = source.length();
    glShaderSource(shaderObj, 1, &sourcePtr, &length);

    return 0;
}

GLint makeShader(std::string vertexFileName, std::string fragmentFileName)
{
    // �V�F�[�_�[�I�u�W�F�N�g�쐬
    GLuint vertShaderObj = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint shader;

    // �V�F�[�_�[�R���p�C���ƃ����N�̌��ʗp�ϐ�
    GLint compiled, linked;

    /* �V�F�[�_�[�̃\�[�X�v���O�����̓ǂݍ��� */
    if (readShaderSource(vertShaderObj, vertexFileName)) return -1;
    if (readShaderSource(fragShaderObj, fragmentFileName)) return -1;

    /* �o�[�e�b�N�X�V�F�[�_�[�̃\�[�X�v���O�����̃R���p�C�� */
    glCompileShader(vertShaderObj);
    glGetShaderiv(vertShaderObj, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in vertex shader.\n");
        return -1;
    }

    /* �t���O�����g�V�F�[�_�[�̃\�[�X�v���O�����̃R���p�C�� */
    glCompileShader(fragShaderObj);
    glGetShaderiv(fragShaderObj, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in fragment shader.\n");
        return -1;
    }

    /* �v���O�����I�u�W�F�N�g�̍쐬 */
    shader = glCreateProgram();

    /* �V�F�[�_�[�I�u�W�F�N�g�̃V�F�[�_�[�v���O�����ւ̓o�^ */
    glAttachShader(shader, vertShaderObj);
    glAttachShader(shader, fragShaderObj);

    /* �V�F�[�_�[�I�u�W�F�N�g�̍폜 */
    glDeleteShader(vertShaderObj);
    glDeleteShader(fragShaderObj);

    /* �V�F�[�_�[�v���O�����̃����N */
    glLinkProgram(shader);
    glGetProgramiv(shader, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE)
    {
        fprintf(stderr, "Link error.\n");
        return -1;
    }

    return shader;
}