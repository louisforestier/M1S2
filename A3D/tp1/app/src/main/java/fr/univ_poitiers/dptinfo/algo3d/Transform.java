package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

public class Transform {

    private float posx;
    private float posy;
    private float posz;
    private float rotx;
    private float roty;
    private float rotz;
    private float scalex;
    private float scaley;
    private float scalez;

    public Transform(float posx, float posy, float posz, float rotx, float roty, float rotz, float scalex, float scaley, float scalez) {
        this.posx = posx;
        this.posy = posy;
        this.posz = posz;
        this.rotx = rotx;
        this.roty = roty;
        this.rotz = rotz;
        this.scalex = scalex;
        this.scaley = scaley;
        this.scalez = scalez;
    }

    public float[] getModelMatrix(){
        float[] modelMatrix = new float[16];
        Matrix.setIdentityM(modelMatrix,0);
        Matrix.rotateM(modelMatrix, 0, rotx, 1.0F, 0.0F, 0.0F);
        Matrix.rotateM(modelMatrix, 0, roty, 0.0F, 1.0F, 0.0F);
        Matrix.rotateM(modelMatrix, 0, rotz, 0.0F, 0.0F, 1.0F);
        Matrix.translateM(modelMatrix,0,this.posx,this.posy,this.posz);
        Matrix.scaleM(modelMatrix,0,scalex,scaley,scalez);
        return modelMatrix;
    }

    public float getPosx() {
        return posx;
    }

    public void setPosx(float posx) {
        this.posx = posx;
    }

    public float getPosy() {
        return posy;
    }

    public void setPosy(float posy) {
        this.posy = posy;
    }

    public float getPosz() {
        return posz;
    }

    public void setPosz(float posz) {
        this.posz = posz;
    }

    public float getRotx() {
        return rotx;
    }

    public void setRotx(float rotx) {
        this.rotx = rotx;
    }

    public float getRoty() {
        return roty;
    }

    public void setRoty(float roty) {
        this.roty = roty;
    }

    public float getRotz() {
        return rotz;
    }

    public void setRotz(float rotz) {
        this.rotz = rotz;
    }

    public float getScalex() {
        return scalex;
    }

    public void setScalex(float scalex) {
        this.scalex = scalex;
    }

    public float getScaley() {
        return scaley;
    }

    public void setScaley(float scaley) {
        this.scaley = scaley;
    }

    public float getScalez() {
        return scalez;
    }

    public void setScalez(float scalez) {
        this.scalez = scalez;
    }
}
