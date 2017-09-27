#version 330 core

in float view_eye_depth;

// Ouput data
layout(location = 0) out vec4 fragmentdepth;


void main(){
  fragmentdepth = vec4(vec3(view_eye_depth), 1);
}
