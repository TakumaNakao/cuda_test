#version 460

//
// shader.vert
//

in vec3 position;

void main(void)
{
    gl_Position = vec4(position, 1.0);
    //gl_FrontColor = vec4(1.0, 0.0, 0.0, 1.0);
}