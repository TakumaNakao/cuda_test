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

// �V�F�[�_�I�u�W�F�N�g�̃R���p�C�����ʂ�\������
static GLboolean printShaderInfoLog(GLuint shader, const char* str)
{
	// �R���p�C�����ʂ��擾����
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) std::cerr << "Compile Error in " << str << std::endl;

	// �V�F�[�_�̃R���p�C�����̃��O�̒������擾����
	GLsizei bufSize;
	glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &bufSize);

	if (bufSize > 1)
	{
		// �V�F�[�_�̃R���p�C�����̃��O�̓��e���擾����
		GLchar* infoLog = new GLchar[bufSize];
		GLsizei length;
		glGetShaderInfoLog(shader, bufSize, &length, infoLog);
		std::cerr << infoLog << std::endl;
		delete[] infoLog;
	}

	return (GLboolean)status;
}

// �v���O�����I�u�W�F�N�g�̃����N���ʂ�\������
static GLboolean printProgramInfoLog(GLuint program)
{
	// �����N���ʂ��擾����
	GLint status;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) std::cerr << "Link Error" << std::endl;

	// �V�F�[�_�̃����N���̃��O�̒������擾����
	GLsizei bufSize;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufSize);

	if (bufSize > 1)
	{
		// �V�F�[�_�̃����N���̃��O�̓��e���擾����
		GLchar* infoLog = new GLchar[bufSize];
		GLsizei length;
		glGetProgramInfoLog(program, bufSize, &length, infoLog);
		std::cerr << infoLog << std::endl;
		delete[] infoLog;
	}

	return (GLboolean)status;
}

// �v���O�����I�u�W�F�N�g�̍쐬
static GLuint createProgram(const char* vsrc, const char* pv, const char* fsrc, const char* fc)
{
	// �o�[�e�b�N�X�V�F�[�_�̃V�F�[�_�I�u�W�F�N�g
	GLuint vobj = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vobj, 1, &vsrc, NULL);
	glCompileShader(vobj);
	printShaderInfoLog(vobj, "vertex shader");

	// �t���O�����g�V�F�[�_�̃V�F�[�_�I�u�W�F�N�g�̍쐬
	GLuint fobj = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fobj, 1, &fsrc, NULL);
	glCompileShader(fobj);
	printShaderInfoLog(fobj, "fragment shader");

	// �V�F�[�_�I�u�W�F�N�g�̎��t��
	GLuint program = glCreateProgram();
	glAttachShader(program, vobj);
	glDeleteShader(vobj);
	glAttachShader(program, fobj);
	glDeleteShader(fobj);

	// �v���O�����I�u�W�F�N�g�̃����N
	glBindAttribLocation(program, 0, pv);
	glBindFragDataLocation(program, 0, fc);
	glLinkProgram(program);
	printProgramInfoLog(program);

	return program;
}