package fr.univ_poitiers.dptinfo.algo3d;


public class Ball extends GameObject{

    static private Mesh sphere= new Sphere(50,50);
    static boolean isInitialized = false;
    public Ball(float radius, float posx, float posz, float[] color) {
        super(color);
        this.setMesh(Ball.sphere);
        this.getTransform().posx(posx).posz(posz).posy(radius).scalex(radius).scaley(radius).scalez(radius);
    }


    @Override
    public void initGraphics() {
        if (!isInitialized){
            super.initGraphics();
            isInitialized = true;
        }
    }


    static public void onPause(){
        isInitialized = false;
    }
}
