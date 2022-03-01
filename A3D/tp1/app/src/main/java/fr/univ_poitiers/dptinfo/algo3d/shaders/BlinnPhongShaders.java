package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

public class BlinnPhongShaders extends LightingShaders{


    /**
     * Fragment shader
     */
    private static final String FRAGSRC=
            "precision mediump float;\n"
                    +"uniform mat4 uModelViewMatrix;\n"
                    +"uniform mat4 uProjectionMatrix;\n"
                    +"uniform mat3 uNormalMatrix;\n"
                    // Light source definition
                    +"uniform float uConstantAttenuation;\n"
                    +"uniform float uLinearAttenuation;\n"
                    +"uniform float uQuadraticAttenuation;\n"
                    +"uniform vec4 uAmbiantLight;\n"
                    +"uniform bool uLighting;\n"
                    +"uniform vec3 uLightPos;\n"
                    +"uniform vec4 uLightColor;\n"
                    // Material definition
                    +"uniform bool uNormalizing;\n"
                    +"uniform vec4 uMaterialColor;\n"
                    //Specular effect
                    +"uniform float uMaterialShininess;\n"
                    +"uniform vec4 uLightSpecular;\n"
                    +"uniform vec4 uMaterialSpecular;\n"

                    +"varying vec4 posf;\n"
                    +"varying vec3 normalf;\n"

                    +"void main(void) {\n"
                    +"  if (uLighting)\n"
                    +"  {\n"
                    +"    float distance = length(uLightPos-posf.xyz);\n"
                    +"    float attenuation = 1.0 /(uConstantAttenuation + uLinearAttenuation* distance + uQuadraticAttenuation * (distance * distance)) ;\n"
                    +"    vec3 viewdir=normalize(-posf.xyz);\n"
                    +"    vec3 normal = normalize(normalf);"
                    +"    vec3 lightdir=normalize(uLightPos-posf.xyz);\n"

                    +"    vec3 halfdir = normalize(lightdir + viewdir);\n"

                    +"    float weight = max(dot(normal, lightdir),0.0);\n"
                    +"    vec4 dColor = uMaterialColor*(uAmbiantLight+weight*uLightColor);\n"

                    +"    float spec = pow(max(dot(halfdir, normal), 0.0), uMaterialShininess*4.0);\n"
                    +"    vec4 specColor = uMaterialSpecular*uLightSpecular*spec;\n"

                    +"    dColor *= attenuation;\n"
                    +"    specColor *= attenuation;\n"
                    +"    gl_FragColor = dColor + specColor;\n"
                    +"  }\n"
                    +"  else gl_FragColor = uMaterialColor;\n"
                    +"}\n";

    /**
     * Vertex shader
     */
    private static final String VERTSRC=
            // Matrices
            "precision mediump float;\n"
                    +"uniform mat4 uModelViewMatrix;\n"
                    +"uniform mat4 uProjectionMatrix;\n"
                    +"uniform mat3 uNormalMatrix;\n"
                    // Light source definition
                    +"uniform vec4 uAmbiantLight;\n"
                    +"uniform bool uLighting;\n"
                    +"uniform vec3 uLightPos;\n"
                    +"uniform vec4 uLightColor;\n"
                    // Material definition
                    +"uniform bool uNormalizing;\n"
                    +"uniform vec4 uMaterialColor;\n"
                    // vertex attributes
                    +"attribute vec3 aVertexPosition;\n"
                    +"attribute vec3 aVertexNormal;\n"
                    // Interpolated data

                    +"varying vec4 posf;\n"
                    +"varying vec3 normalf;\n"

                    +"void main(void) {\n"
                    +"  posf=uModelViewMatrix*vec4(aVertexPosition, 1.0);\n"
                    +"  normalf=uNormalMatrix * aVertexNormal;\n"
                    +"  if (uNormalizing) normalf=normalize(normalf);\n"
                    +"  gl_Position= uProjectionMatrix*posf;\n"
                    +"}\n";

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */
    public BlinnPhongShaders(Context context) {
        super(context);
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"blinn_phong_vert.glsl","blinn_phong_frag.glsl");
    }

}
