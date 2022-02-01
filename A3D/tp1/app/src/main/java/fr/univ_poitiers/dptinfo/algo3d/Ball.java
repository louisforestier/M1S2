package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

public class Ball extends GameObject{

    static private Mesh sphere= new Sphere(5);

    public Ball(float radius, float posx, float posz, float[] color) {
        super(color);
        this.setMesh(Ball.sphere);
        this.getTransform().posx(posx).posz(posz).posy(radius).scalex(radius).scaley(radius).scalez(radius);
    }
}
