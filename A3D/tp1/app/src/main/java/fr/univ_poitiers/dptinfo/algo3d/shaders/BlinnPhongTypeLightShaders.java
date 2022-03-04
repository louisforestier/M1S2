package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;
import android.opengl.GLES20;

import fr.univ_poitiers.dptinfo.algo3d.MainActivity;

public class BlinnPhongTypeLightShaders extends LightingShaders{

    protected int uLightType;

    protected int uSpotDirection;

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */
    public BlinnPhongTypeLightShaders(Context context) {
        super(context);
    }

    @Override
    public void findVariables() {
        super.findVariables();
        this.uLightType = GLES20.glGetUniformLocation(this.shaderprogram, "uLightType");
        if (this.uLightType==-1) MainActivity.log("Warning:  uLightType not found in shaders...");
        this.uSpotDirection = GLES20.glGetUniformLocation(this.shaderprogram, "uSpotDir");
        if (this.uSpotDirection==-1) throw new RuntimeException("uSpotDir not found in shaders");
    }

    public void setLightType(int lightType) {
        GLES20.glUniform1i(this.uLightType,lightType);
    }

    public void setSpotDirection(final float[] spotdir)
    {
        GLES20.glUniform3fv(this.uSpotDirection,1,spotdir,0);
    }


    @Override
    public boolean useTypeLight() {
        return true;
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"blinn_phong_with_type_light_vert.glsl","blinn_phong_with_type_light_frag.glsl");
    }

}
