package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.Matrix;

public class Ball extends GameObject{

    static private Sphere sphere= new Sphere(5);
    static boolean isInitialized = false;

    public Ball(float radius, float posx, float posz, float[] color) {
        super(color);
        this.setMesh(Ball.sphere);
        TransformBuilder tb = new TransformBuilder();
        this.setTransform(tb.posx(posx).posz(posz).posy(radius).scalex(radius).scaley(radius).scalez(radius).buildTransform());
    }

    @Override
    public void initGraphics() {
        if (!Ball.isInitialized){
            super.initGraphics();
            Ball.isInitialized = true;
        }
    }
}
