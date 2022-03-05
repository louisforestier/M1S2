package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;
import android.opengl.GLES20;

import fr.univ_poitiers.dptinfo.algo3d.Light;
import fr.univ_poitiers.dptinfo.algo3d.MainActivity;

public class BlinnPhongMultipleLightShaders extends MultipleLightingShaders{

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */
    public BlinnPhongMultipleLightShaders(Context context) {
        super(context);
    }

    @Override
    public void findVariables() {
        super.findVariables();
    }

    @Override
    public boolean useTypeLight() {
        return true;
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"blinn_phong_with_multiple_lights_vert.glsl","blinn_phong_with_multiple_lights_frag.glsl");
    }

    public void setDirLight(Light light){
        GLES20.glUniform3fv(GLES20.glGetUniformLocation(this.shaderprogram,"dirLight.direction"),1,light.getDirection(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"dirLight.ambient"),1,light.getAmbient(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"dirLight.diffuse"),1,light.getDiffuse(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"dirLight.specular"),1,light.getSpecular(),0);
    }

    public void setPointLight(Light light) {
        GLES20.glUniform3fv(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.position"),1,light.getPosition(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.ambient"),1,light.getAmbient(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.diffuse"),1,light.getDiffuse(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.specular"),1,light.getSpecular(),0);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.constant"),light.getConstant());
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.linear"),light.getLinear());
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"pointLight.quadratic"),light.getQuadratic());
    }

    public void setSpotLight(Light light) {
        GLES20.glUniform3fv(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.position"),1,light.getPosition(),0);
        GLES20.glUniform3fv(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.direction"),1,light.getDirection(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.ambient"),1,light.getAmbient(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.diffuse"),1,light.getDiffuse(),0);
        GLES20.glUniform4fv(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.specular"),1,light.getSpecular(),0);
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.cutOff"),light.getCutOff());
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.outerCutOff"),light.getOuterCutOff());
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.constant"),light.getConstant());
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.linear"),light.getLinear());
        GLES20.glUniform1f(GLES20.glGetUniformLocation(this.shaderprogram,"spotLight.quadratic"),light.getQuadratic());
    }
}
