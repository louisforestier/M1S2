#version 100
precision mediump float;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;
// Light source definition
uniform float uConstantAttenuation;
uniform float uLinearAttenuation;
uniform float uQuadraticAttenuation;
uniform vec4 uAmbiantLight;
uniform bool uLighting;
uniform vec3 uLightPos;
uniform vec4 uLightColor;
// Material definition
uniform bool uNormalizing;
uniform vec4 uMaterialColor;
//Specular effect
uniform float uMaterialShininess;
uniform vec4 uLightSpecular;
uniform vec4 uMaterialSpecular;

varying vec4 posf;
varying vec3 normalf;

void main(void) {
  if (uLighting)
  {
    float distance = length(uLightPos-posf.xyz);
    float attenuation = 1.0 /(uConstantAttenuation + uLinearAttenuation* distance + uQuadraticAttenuation * (distance * distance)) ;
    vec3 viewdir=normalize(-posf.xyz);
    vec3 normal = normalize(normalf);
    vec3 lightdir=normalize(uLightPos-posf.xyz);

    vec3 halfdir = normalize(lightdir + viewdir);

    float weight = max(dot(normal, lightdir),0.0);
    vec4 dColor = uMaterialColor*(uAmbiantLight+weight*uLightColor);

    float spec = pow(max(dot(halfdir, normal), 0.0), uMaterialShininess*4.0);
    vec4 specColor = uMaterialSpecular*uLightSpecular*spec;

    dColor *= attenuation;
    specColor *= attenuation;
    gl_FragColor = dColor + specColor;
  }
  else gl_FragColor = uMaterialColor;
}
