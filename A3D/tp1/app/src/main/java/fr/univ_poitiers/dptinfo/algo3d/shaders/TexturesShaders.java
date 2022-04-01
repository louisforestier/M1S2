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
}
