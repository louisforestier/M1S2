package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class ShadowShaders extends MultipleLightingShaders{


    protected int aVertexTexture;
    protected int uTextureUnit;
    protected int uTexturing;
    private int lightSpaceMatrix;
    private int uModelMatrix;
    private int shadowMap;
    private int poissonDisk;


    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */

    public ShadowShaders(Context context) {
        super(context);
    }

    @Override
    public void findVariables() {
        super.findVariables();
        this.aVertexTexture = GLES20.glGetAttribLocation(this.shaderprogram, "aVertexTexture");
        if (this.aVertexTexture==-1) throw new RuntimeException("aVertexTexture not found in shaders");
        GLES20.glEnableVertexAttribArray(this.aVertexTexture);
        this.uTextureUnit = GLES20.glGetUniformLocation(this.shaderprogram, "uTextureUnit");
        if (this.uTextureUnit==-1) throw new RuntimeException("uTextureUnit not found in shaders");
        this.uTexturing = GLES20.glGetUniformLocation(this.shaderprogram, "uTexturing");
        if (this.uTexturing==-1) throw new RuntimeException("uTexturing not found in shaders");

        this.lightSpaceMatrix = GLES20.glGetUniformLocation(this.shaderprogram, "lightSpaceMatrix");
        if (this.lightSpaceMatrix==-1) throw new RuntimeException("lightSpaceMatrix not found in shaders");

        this.uModelMatrix = GLES20.glGetUniformLocation(this.shaderprogram, "uModelMatrix");
        if (this.uModelMatrix==-1) throw new RuntimeException("uModelMatrix not found in shaders");

        this.shadowMap = GLES20.glGetUniformLocation(this.shaderprogram, "shadowMap");
        if (this.shadowMap==-1) throw new RuntimeException("shadowMap not found in shaders");

        this.poissonDisk = GLES20.glGetUniformLocation(this.shaderprogram, "poissonDisk");
        if (this.poissonDisk==-1) throw new RuntimeException("poissonDisk not found in shaders");
        setPoissonDisk();
    }

    public void setPoissonDisk(){
        float[][] poissondisk = new float[][]{
                {-0.94201624f, -0.39906216f},
                {0.94558609f, -0.76890725f},
                {-0.094184101f, -0.92938870f},
                {0.34495938f, 0.29387760f},
                {-0.91588581f, 0.45771432f},
                {-0.81544232f, -0.87912464f},
                {-0.38277543f, 0.27676845f},
                {0.97484398f, 0.75648379f},
                {0.44323325f, -0.97511554f},
                {0.53742981f, -0.47373420f},
                {-0.26496911f, -0.41893023f},
                {0.79197514f, 0.19090188f},
                {-0.24188840f, 0.99706507f},
                {-0.81409955f, 0.91437590f},
                {0.19984126f, 0.78641367f},
                {0.14383161f, -0.14100790f}
        };
        ByteBuffer poissonbytebuff = ByteBuffer.allocateDirect(poissondisk.length * 2 * Float.BYTES );
        poissonbytebuff.order(ByteOrder.nativeOrder());
        FloatBuffer poissonbuff = poissonbytebuff.asFloatBuffer();
        for (int i = 0 ; i < 16 ; i++){
            for (int j = 0 ; j < 2 ; j++){
                poissonbuff.put(poissondisk[i][j]);
            }
        }
        poissonbuff.position(0);
        GLES20.glUniform2fv(poissonDisk,16,poissonbuff);
    }

    @Override
    public boolean useTypeLight() {
        return true;
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"shadow_vert.glsl","shadow_frag.glsl");
    }

    @Override
    public void setTexturePointer(int size,int dtype)
    {
        GLES20.glVertexAttribPointer(this.aVertexTexture, size, dtype, false, 0, 0);
    }

    @Override
    public void setTextureUnit(final int textureUnit)
    {
        GLES20.glUniform1i(this.uTextureUnit,textureUnit);
    }

    @Override
    public void setTexturing(final boolean state){
        if (this.uTexturing!=-1) GLES20.glUniform1i(this.uTexturing,state?1:0);
    }

    @Override
    public void setLightSpaceMatrix(float[] matrix) {
        GLES20.glUniformMatrix4fv(this.lightSpaceMatrix, 1, false, matrix,0);
    }

    @Override
    public void setModelMatrix(float[] matrix){
        GLES20.glUniformMatrix4fv(this.uModelMatrix, 1, false, matrix,0);
    }

    @Override
    public void setDepthMap(int depthMap) {
        GLES20.glUniform1i(this.shadowMap,depthMap);
    }

}
