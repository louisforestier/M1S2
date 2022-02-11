package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

public class Transform {

    private Vec3f pos;
    private Vec3f rot;
    private Vec3f scale;

    public Transform() {
        pos = new Vec3f();
        rot = new Vec3f();
        scale = new Vec3f(1.0f,1.0f,1.0f);
    }

    public float[] getModelMatrix(){
        float[] modelMatrix = new float[16];
        Matrix.setIdentityM(modelMatrix,0);
        Matrix.translateM(modelMatrix,0,pos.x,pos.y,pos.z);
        Matrix.rotateM(modelMatrix, 0, rot.z, 0.0F, 0.0F, 1.0F);
        Matrix.rotateM(modelMatrix, 0, rot.x, 1.0F, 0.0F, 0.0F);
        Matrix.rotateM(modelMatrix, 0, rot.y, 0.0F, 1.0F, 0.0F);
        Matrix.scaleM(modelMatrix,0,scale.x,scale.y,scale.z);
        return modelMatrix;
    }

    public float getPosx() {
        return pos.x;
    }

    public Transform posx(float posx) {
        this.pos.x = posx;
        return this;
    }

    public float getPosy() {
        return pos.y;
    }

    public Transform posy(float posy) {
        this.pos.y = posy;
        return this;
    }

    public float getPosz() {
        return pos.z;
    }

    public Transform posz(float posz) {
        this.pos.z = posz;
        return this;
    }

    public float getRotx() {
        return rot.x;
    }

    public Transform rotx(float rotx) {
        this.rot.x = rotx;
        return this;
    }

    public float getRoty() {
        return rot.y;
    }

    public Transform roty(float roty) {
        this.rot.y = roty;
        return this;
    }

    public float getRotz() {
        return rot.z;
    }

    public Transform rotz(float rotz) {
        this.rot.z = rotz;
        return this;
    }

    public float getScalex() {
        return scale.x;
    }

    public Transform scalex(float scalex) {
        this.scale.x = scalex;
        return this;
    }

    public float getScaley() {
        return scale.y;
    }

    public Transform scaley(float scaley) {
        this.scale.y = scaley;
        return this;
    }

    public float getScalez() {
        return scale.z;
    }

    public Transform scalez(float scalez) {
        this.scale.z = scalez;
        return this;
    }
}
