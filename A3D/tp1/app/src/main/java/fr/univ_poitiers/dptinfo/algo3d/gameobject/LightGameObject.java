package fr.univ_poitiers.dptinfo.algo3d.gameobject;

import android.opengl.Matrix;

import fr.univ_poitiers.dptinfo.algo3d.Light;
import fr.univ_poitiers.dptinfo.algo3d.LightType;
import fr.univ_poitiers.dptinfo.algo3d.shaders.BasicShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.BlinnPhongMultipleLightShaders;
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

    public float[] getDir(final float[] viewmatrix){
        float[] lightDir = new float[4];
        float[] lightlocalDir = new float[]{
                (float) (Math.cos(Math.toRadians(getTransform().getRoty())) * Math.cos(Math.toRadians(getTransform().getRotx()))),
                (float) Math.sin(Math.toRadians(getTransform().getRotx())),
                (float) (Math.sin(Math.toRadians(getTransform().getRoty())) * Math.cos(Math.toRadians(getTransform().getRotx()))),
                0.f
        };
        Matrix.multiplyMV(lightDir,0,viewmatrix,0,lightlocalDir,0);
        return new float[]{lightDir[0],lightDir[1],lightDir[2]};
    }


    public void initLighting(BasicShaders shaders, final float[] modelviewmatrix) {
        if (shaders.useTypeLight()){
            light.setPosition(getPos(modelviewmatrix));
            light.setDirection(getDir(modelviewmatrix));
            switch (light.getType()) {
                case DIRECTIONAL:
                    ((BlinnPhongMultipleLightShaders) shaders).setDirLight(light);
                    break;
                case POINT:
                    ((BlinnPhongMultipleLightShaders) shaders).setPointLight(light);
                    break;
                case SPOT:
                    ((BlinnPhongMultipleLightShaders) shaders).setSpotLight(light);
                    break;
            }
        }
    }
}
