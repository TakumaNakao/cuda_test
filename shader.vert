#version 460

//
// shader.vert
//

uniform mat4 MVP;
in vec3 position;

void main(void)
{
    gl_Position = MVP * vec4(position, 1.0);
    //gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);
}