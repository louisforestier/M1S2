package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

import java.util.ArrayList;
import java.util.List;

public class GameObject {

    private Mesh mesh;
    private Transform transform;
    private float[] color;
    private List<GameObject> children = new ArrayList<>();
    private GameObject parent = null;


    public GameObject(float[] color){
        this.transform = new Transform();
        this.color = color;
    }

    public GameObject() {
        this.transform = new Transform();
        this.color = MyGLRenderer.white;
    }

    public void setMesh(Mesh mesh) {
        this.mesh = mesh;
    }

    public Transform getTransform() {
        return transform;
    }

    public void addChildren(GameObject child){
        this.children.add(child);
        child.parent = this;
    }

    public void initGraphics(){
        if (this.mesh != null)
            this.mesh.initGraphics();
        if (this.children.size() > 0){
            for (GameObject go : this.children) {
                go.initGraphics();
            }
        }
    }

    public void draw(NoLightShaders shaders, final float[] viewmatrix){
        float[] modelviewmatrix = new float[16];
        Matrix.multiplyMM(modelviewmatrix,0,viewmatrix,0,transform.getModelMatrix(),0);
        shaders.setModelViewMatrix(modelviewmatrix);
        shaders.setColor(color);
        if (this.mesh != null)
            mesh.draw(shaders);
        if (this.children.size() > 0){
            for (GameObject go : this.children) {
                go.draw(shaders,modelviewmatrix);
            }
        }
    }
    public void draw(NoLightShaders shaders, final float[] viewmatrix, DrawMode drawMode){
        float[] modelviewmatrix = new float[16];
        Matrix.multiplyMM(modelviewmatrix,0,viewmatrix,0,transform.getModelMatrix(),0);
        shaders.setModelViewMatrix(modelviewmatrix);
        shaders.setColor(color);
        if (this.mesh != null)
            mesh.draw(shaders,drawMode);
        if (this.children.size() > 0){
            for (GameObject go : this.children) {
                go.draw(shaders,modelviewmatrix,drawMode);
            }
        }
    }

}
