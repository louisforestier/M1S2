package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

public class Ball {

    private float radius;
    private float posx;
    private float posz;
    private float[] color;
    static private Sphere sphere= new Sphere(5);
    static {
        sphere.initGraphics();
    }

    private float[] modelmatrix;

    public Ball(float radius, float posx, float posz, float[] color) {
        this.radius = radius;
        this.posx = posx;
        this.posz = posz;
        this.color = color;
        modelmatrix = new float[16];
        Matrix.setIdentityM(modelmatrix,0);
        Matrix.translateM(modelmatrix,0,-this.posx,this.radius,-this.posz);
        Matrix.scaleM(modelmatrix,0,this.radius,this.radius,this.radius);
    }

    public void draw(final NoLightShaders shaders,final float[] viewmatrix){
        float[] modelviewmatrix = new float[16];
        Matrix.multiplyMM(modelviewmatrix,0,viewmatrix,0,modelmatrix,0);
        shaders.setModelViewMatrix(modelviewmatrix);
        shaders.setColor(color);
        sphere.draw(shaders);
    }
}
