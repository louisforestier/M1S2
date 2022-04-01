package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;
import android.opengl.GLES20;

import fr.univ_poitiers.dptinfo.algo3d.MainActivity;

public class BlinnPhongTypeLightShaders extends LightingShaders {

    protected int uLightType;

    protected int uSpotDirection;
    private int uCutOff;
    private int uOuterCutOff;

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
        if (this.uLightType == -1) MainActivity.log("Warning:  uLightType not found in shaders...");
        this.uCutOff = GLES20.glGetUniformLocation(this.shaderprogram, "uCutOff");
        if (this.uCutOff == -1) MainActivity.log("Warning:  uCutOff not found in shaders...");
        this.uOuterCutOff = GLES20.glGetUniformLocation(this.shaderprogram, "uOuterCutOff");
        if (this.uOuterCutOff == -1)
            MainActivity.log("Warning:  uOuterCutOff not found in shaders...");
        this.uSpotDirection = GLES20.glGetUniformLocation(this.shaderprogram, "uSpotDir");
        if (this.uSpotDirection == -1) throw new RuntimeException("uSpotDir not found in shaders");
    }

    public void setLightType(int lightType) {
        GLES20.glUniform1i(this.uLightType, lightType);
    }

    public void setSpotDirection(final float[] spotdir) {
        GLES20.glUniform3fv(this.uSpotDirection, 1, spotdir, 0);
    }


    @Override
    public boolean useTypeLight() {
        return true;
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context, "blinn_phong_with_type_light_vert.glsl", "blinn_phong_with_type_light_frag.glsl");
    }

    public void setCutOff(float cutOff) {
        GLES20.glUniform1f(this.uCutOff, cutOff);
    }

    public void setOuterCutOff(float outerCutOff) {
        GLES20.glUniform1f(this.uOuterCutOff, outerCutOff);
    }

}
