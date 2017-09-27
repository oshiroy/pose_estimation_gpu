#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;

// Values that stay constant for the whole mesh.
uniform mat4 MVP;
uniform mat4 MV;

out float view_eye_depth;

void main(){
	gl_Position =  MVP * vec4(vertexPosition_modelspace, 1);
  vec3 view_eye_pos = (MV * vec4(vertexPosition_modelspace, 1)).xyz;

  view_eye_depth  = - view_eye_pos.z;
}