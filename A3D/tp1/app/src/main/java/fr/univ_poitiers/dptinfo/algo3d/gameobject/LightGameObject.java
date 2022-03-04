package fr.univ_poitiers.dptinfo.algo3d.gameobject;

import android.opengl.Matrix;

import fr.univ_poitiers.dptinfo.algo3d.Light;
import fr.univ_poitiers.dptinfo.algo3d.LightType;
import fr.univ_poitiers.dptinfo.algo3d.shaders.BlinnPhongTypeLightShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.LightingShaders;

public class LightGameObject extends GameObject{
    private Light light;

    public LightGameObject(LightType lightType){
        light = new Light(lightType);
    }
    public LightGameObject(Light light) {
        this.light = light;
    }

    public Light getLight() {
        return light;
    }

    public void setLight(Light light) {
        this.light = light;
    }

    public float[] getPos(final float[] viewmatrix){
        float[] lightPos = new float[4];
        Matrix.multiplyMV(lightPos,0,viewmatrix,0,new float[]{getTransform().getPosx(),getTransform().getPosy(),getTransform().getPosz(),1.0f},0);
        return new float[]{lightPos[0],lightPos[1],lightPos[2]};
    }


    public void initLighting(LightingShaders shaders, final float[] modelviewmatrix) {
        if (shaders.useTypeLight()){
            ((BlinnPhongTypeLightShaders) shaders).setLightType(light.getType().getValue());
            if (light.getType() == LightType.SPOT){
                ((BlinnPhongTypeLightShaders) shaders).setSpotDirection(new float[3]);
            }
        }
        shaders.setLightPosition(this.getPos(modelviewmatrix));
        shaders.setLightColor(light.getDiffuse());
        shaders.setAmbiantLight(light.getAmbient());
        shaders.setLightSpecular(light.getSpecular());
        shaders.setLightAttenuation(light.getConstant(),light.getLinear(), light.getQuadratic());
    }
}
