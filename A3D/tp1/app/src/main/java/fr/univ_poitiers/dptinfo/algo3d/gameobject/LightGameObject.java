package fr.univ_poitiers.dptinfo.algo3d.gameobject;

import android.opengl.Matrix;

import fr.univ_poitiers.dptinfo.algo3d.Light;

public class LightGameObject extends GameObject{
    private Light light;

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
}
