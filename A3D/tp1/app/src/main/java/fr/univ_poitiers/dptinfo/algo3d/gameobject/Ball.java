package fr.univ_poitiers.dptinfo.algo3d.gameobject;


import fr.univ_poitiers.dptinfo.algo3d.mesh.Material;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Mesh;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Sphere;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;

public class Ball extends GameObject {

    static private Mesh sphere = new Sphere(50, 50);
    static boolean isInitialized = false;

    public Ball(float radius, float posx, float posz, Material material) {
        super();
        this.setMesh(Ball.sphere);
        this.addMeshRenderer(material);
        this.getTransform().posx(posx).posz(posz).posy(radius).scalex(radius).scaley(radius).scalez(radius);
    }


    @Override
    public void start() {
        if (!isInitialized) {
            super.start();
            isInitialized = true;
        }
    }


    static public void onPause() {
        isInitialized = false;
    }
}
