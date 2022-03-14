#version 100
#define POINT_LIGHT 0
#define DIRECTIONAL_LIGHT 1
#define SPOT_LIGHT 2
precision mediump float;
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat3 uNormalMatrix;
// Light source definition
uniform bool uLighting;
// Material definition
uniform bool uNormalizing;
uniform vec4 uMaterialColor;
//Specular effect
uniform float uMaterialShininess;
uniform vec4 uLightSpecular;
uniform vec4 uMaterialSpecular;

uniform sampler2D uTextureUnit;
uniform sampler2D shadowMap;
uniform bool uTexturing;

varying vec4 posf;
varying vec3 normalf;
varying vec2 texturef;
varying vec4 lightspaceposf;


struct DirLight {
    vec3 direction;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;

    float constant;
    float linear;
    float quadratic;

    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

#define NB_DIR_LIGHTS 1
#define NB_SPOT_LIGHTS 1
#define NB_POINT_LIGHTS 1

uniform DirLight dirLights[NB_DIR_LIGHTS];
uniform PointLight pointLights[NB_POINT_LIGHTS];
uniform SpotLight spotLights[NB_SPOT_LIGHTS];

uniform vec2 poissonDisk[16];


// Returns a random number based on a vec3 and an int.
float random(vec3 seed, int i){
    vec4 seed4 = vec4(seed, i);
    float dot_product = dot(seed4, vec4(12.9898, 78.233, 45.164, 94.673));
    return fract(sin(dot_product) * 43758.5453);
}

float shadowCalculation(vec4 lightspaceposf, vec3 normal, vec3 lightdir)
{
    vec3 projCoords = lightspaceposf.xyz / lightspaceposf.w;
    projCoords = projCoords * 0.5 + 0.5;
    float closestDepth = texture2D(shadowMap, projCoords.xy).r;
    float currentDepth = projCoords.z;
    float cosTheta = clamp(dot(normal, lightdir), 0.0, 1.0);
    float bias = 0.0009*tan(acos(cosTheta));
    bias = clamp(bias, 0.0, 0.01);
    float shadow = 0.0;
    for (int i=0;i<4;i++){
        // use either :
        //  - Always the same samples.
        //    Gives a fixed pattern in the shadow, but no noise
        int index = i;
        //  - A random sample, based on the pixel's screen location.
        //    No banding, but the shadow moves with the camera, which looks weird.
        //int index = int(mod(16.0*random(gl_FragCoord.xyz, i),16.0));
        //  - A random sample, based on the pixel's position in world space.
        //    The position is rounded to the millimeter to avoid too much aliasing
        //int index = int(mod(16.0 * random(floor(posf.xyz*1000.0), i),16.0));

        // being fully in the shadow will eat up 4*0.2 = 0.8
        // 0.2 potentially remain, which is quite dark.
        float closestDepth = texture2D(shadowMap, vec2(projCoords.xy+poissonDisk[index]/700.0)).r;
        float currentDepth = projCoords.z;
        shadow += currentDepth - bias > closestDepth ? 0.2 * (1.0 - closestDepth) : 0.0;
    }
    //float shadow = currentDepth - bias > closestDepth  ? 1.0 : 0.0;
    return shadow;
}

vec4 calcPointLight(PointLight light, vec3 normal, vec3 posf, vec3 viewdir)
{
    vec4 color;
    if (uTexturing)
    color = texture2D(uTextureUnit, texturef);
    else
    color = vec4(1, 1, 1, 1);
    float distance = length(light.position-posf);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    vec3 lightdir = normalize(light.position-posf);
    vec3 halfdir = normalize(lightdir + viewdir);
    float weight = max(dot(normal, lightdir), 0.0);
    vec4 dColor = color * uMaterialColor * (light.ambient + weight*light.diffuse);
    float spec = pow(max(dot(halfdir, normal), 0.0), uMaterialShininess*4.0);
    vec4 specColor = uMaterialSpecular * light.specular * spec;
    dColor *= attenuation;
    specColor *= attenuation;
    return dColor+specColor;
}

vec4 calcDirLight(DirLight light, vec3 normal, vec3 viewdir)
{
    vec4 color;
    if (uTexturing)
    color = texture2D(uTextureUnit, texturef);
    else
    color = vec4(1, 1, 1, 1);
    vec3 lightdir = normalize(-light.direction);
    vec3 halfdir = normalize(lightdir + viewdir);
    float weight = max(dot(normal, lightdir), 0.0);
    float shadow = shadowCalculation(lightspaceposf, normal, lightdir);
    vec4 dColor = color*uMaterialColor * (light.ambient + (1.0 - shadow) * weight*light.diffuse);
    float spec = pow(max(dot(halfdir, normal), 0.0), uMaterialShininess*4.0);
    vec4 specColor = (1.0 - shadow) * uMaterialSpecular * light.specular * spec;
    return dColor+specColor;
}

vec4 calcSpotLight(SpotLight light, vec3 normal, vec3 posf, vec3 viewdir)
{
    vec4 color;
    if (uTexturing)
    color = texture2D(uTextureUnit, texturef);
    else
    color = vec4(1, 1, 1, 1);
    float distance = length(light.position-posf);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    vec3 lightdir = normalize(light.position-posf);
    float theta = dot(lightdir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    vec3 halfdir = normalize(lightdir + viewdir);
    float weight = max(dot(normal, lightdir), 0.0);
    vec4 dColor = color*uMaterialColor * (light.ambient + weight*light.diffuse*intensity);
    float spec = pow(max(dot(halfdir, normal), 0.0), uMaterialShininess*4.0);
    vec4 specColor = uMaterialSpecular * light.specular * spec;
    dColor *= attenuation;
    specColor *= attenuation * intensity;
    return dColor+specColor;
}


void main(void) {
    if (uLighting)
    {
        vec3 normal = normalize(normalf);
        vec3 viewdir=normalize(-posf.xyz);
        vec4 result;

        for (int i = 0; i < NB_DIR_LIGHTS; i++)
        {
            result += calcDirLight(dirLights[i], normal, viewdir);
        }
        for (int i = 0; i < NB_POINT_LIGHTS; i++)
        {
            //result += calcPointLight(pointLights[i],normal,posf.xyz,viewdir);
        }
        for (int i = 0; i < NB_SPOT_LIGHTS; i++)
        {
            //result += calcSpotLight(spotLights[i],normal,posf.xyz,viewdir);
        }
        gl_FragColor = result;
    }
    else gl_FragColor = uMaterialColor;
}
