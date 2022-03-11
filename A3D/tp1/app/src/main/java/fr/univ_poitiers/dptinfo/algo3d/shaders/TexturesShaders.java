package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;
import android.opengl.GLES20;

public class TexturesShaders extends MultipleLightingShaders{


    protected int aVertexTexture;
    protected int uTextureUnit;
    protected int uTexturing;


    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */

    public TexturesShaders(Context context) {
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


    }

    @Override
    public boolean useTypeLight() {
        return true;
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"texture_vert.glsl","texture_frag.glsl");
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

    public void prepareDepthMap(){
        int[] depthMaptFBO = new int[1];
        GLES20.glGenFramebuffers(1,depthMaptFBO,0);
        int[] depthMap = new int[1];
        final int SHADOW_WIDTH = 1024, SHADOW_HEIGHT = 1024;
        GLES20.glGenTextures(1,depthMap,0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,depthMap[0]);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D,0,GLES20.GL_DEPTH_COMPONENT,SHADOW_WIDTH,SHADOW_HEIGHT,0,GLES20.GL_DEPTH_COMPONENT,GLES20.GL_FLOAT,null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_REPEAT);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_REPEAT);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,depthMaptFBO[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER,GLES20.GL_DEPTH_ATTACHMENT,GLES20.GL_TEXTURE_2D,depthMap[0],0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,0);


        GLES20.glViewport(0,0,SHADOW_WIDTH,SHADOW_HEIGHT);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER,depthMaptFBO[0]);
        GLES20.glClear(GLES20.GL_DEPTH_BUFFER_BIT);
    }
}
