#version 100
precision mediump float;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
attribute vec3 aVertexPosition;
// Interpolated data

void main(void) {
  posf=uModelViewMatrix*vec4(aVertexPosition, 1.0);
  gl_Position= uProjectionMatrix*posf;
}
