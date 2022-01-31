package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

public class GameObject {

    private Mesh mesh;
    private Transform transform;
    private float[] color;

    public GameObject() {
    }

    public GameObject(float[] color){
        this.color = color;
    }

    public void setMesh(Mesh mesh) {
        this.mesh = mesh;
    }

    public void setTransform(Transform transform) {
        this.transform = transform;
    }

    public void initGraphics(){
        this.mesh.initGraphics();
    }

    public void draw(NoLightShaders shaders, final float[] viewmatrix){
        float[] modelviewmatrix = new float[16];
        Matrix.multiplyMM(modelviewmatrix,0,viewmatrix,0,transform.getModelMatrix(),0);
        shaders.setModelViewMatrix(modelviewmatrix);
        shaders.setColor(color);
        mesh.draw(shaders);
    }
}
