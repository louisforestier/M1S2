package fr.univ_poitiers.dptinfo.algo3d;

public class PixelShaders extends LightingShaders{

    /**
     * Fragment shader
     */
    private static final String FRAGSRC=
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

                    +"varying vec4 posf;\n"
                    +"varying vec3 normalf;\n"

                    +"void main(void) {\n"
                    +"  if (uLighting)\n"
                    +"  {\n"
                    +"    vec3 normal = normalize(normalf);"
                    +"    vec3 lightdir=normalize(uLightPos-posf.xyz);\n"
                    +"    float weight = max(dot(normal, lightdir),0.0);\n"
                    +"    gl_FragColor = uMaterialColor*(uAmbiantLight+weight*uLightColor);\n"
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
     * @param renderer
     */
    public PixelShaders(MyGLRenderer renderer) {
        super(renderer);
    }


    @Override
    public int createProgram() {
        return initializeShaders(VERTSRC,FRAGSRC);
    }
}
