package fr.univ_poitiers.dptinfo.algo3d;


import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import fr.univ_poitiers.dptinfo.algo3d.gameobject.Ball;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.GameObject;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.Room;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Cube;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Cylinder;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Donut;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Frustum;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Material;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Pipe;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Pyramid;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Tictac;
import fr.univ_poitiers.dptinfo.algo3d.objimporter.OBJImporter;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShadingMode;

/**
 * Class to represent the scene. It includes all the objects to display, in this case a room
 *
 * @author Philippe Meseure
 * @version 1.0
 */
public class Scene {
    private final GameObject light2;
    private final GameObject light3;

    private List<GameObject> gameObjects = new ArrayList<>();
    private List<GameObject> lights = new ArrayList<GameObject>();

    private GameObject light;
    /**
     * An angle used to animate the viewer
     */
    float anglex, angley;

    float posx, posz;
    float dx,dy,dx2,dy2;


    /**
     * Constructor : build each wall, the floor and the ceiling as quads
     */
    public Scene(Context current) {
        /**
         * A constant for the size of the wall
         */
        // Init observer's view angles
        angley = 0.F;
        Material ceilingMaterial = new Material(MyGLRenderer.darkgray);
        Material wallMaterial = new Material(MyGLRenderer.lightgray);
        GameObject room = new Room(new boolean[]{false, false, true, true}, 6.F, 6.F, 2.5F, new Material(MyGLRenderer.green), ceilingMaterial,wallMaterial);
        gameObjects.add(room);
        GameObject room2 = new Room(new boolean[]{true, false, false, false}, 6.F, 16.F, 2.5F, new Material(MyGLRenderer.red), ceilingMaterial,wallMaterial);
        room2.getTransform().posz(6);
        gameObjects.add(room2);
        GameObject room3 = new Room(new boolean[]{true, true, true, true}, 6.f, 6.f, 2.5f, new Material(MyGLRenderer.cyan), ceilingMaterial,wallMaterial);
        room3.getTransform().posx(6);
        gameObjects.add(room3);
        GameObject room4 = new Room(new boolean[]{false, true, false, true}, 6.f, 6.f, 4.5f, new Material(MyGLRenderer.orange), ceilingMaterial, wallMaterial);
        //je pourrais aussi créer mes portes sur les autres murs mais c'est pour vérifier que la rotation fonctionne correctement
        room4.getTransform().posx(6).posz(-6).roty(90).rotx(180).posy(2.f);
        gameObjects.add(room4);

        GameObject ball = new Ball(1.2f, 1.5f, 1.5f, new Material(MyGLRenderer.orange));
        gameObjects.add(ball);

        GameObject ball2 = new Ball(0.3f, -1.5f, 1.5f, new Material(MyGLRenderer.gray));
        gameObjects.add(ball2);


        InputStream stream = current.getResources().openRawResource(R.raw.armadillo);
        Material armadilloMaterial = new Material(MyGLRenderer.lightgray);
        GameObject armadillo = new GameObject();
        armadillo.setMesh(OBJImporter.importOBJ(stream, ShadingMode.SMOOTH_SHADING));
        armadillo.getTransform().posy(1.F).scalex(0.01F).scaley(0.01F).scalez(0.01F).posx(7.5f);
        armadillo.addMeshRenderer(armadilloMaterial);
        gameObjects.add(armadillo);

        GameObject armadillo2 = new GameObject();
        stream = current.getResources().openRawResource(R.raw.armadillo_with_normals);
        armadillo2.setMesh(OBJImporter.importOBJ(stream,ShadingMode.SMOOTH_SHADING));
        armadillo2.getTransform().posy(1.F).scalex(0.01F).scaley(0.01F).scalez(0.01F).posx(7.5f).posz(1.f);
        armadillo2.addMeshRenderer(armadilloMaterial);
        gameObjects.add(armadillo2);

        stream = current.getResources().openRawResource(R.raw.xyzrgb_dragon);
        GameObject dragon = new GameObject();
        dragon.setMesh(OBJImporter.importOBJ(stream, ShadingMode.FLAT_SHADING));
        dragon.getTransform().posy(1.f).scalex(0.02f).scaley(0.02f).scalez(0.02f).posx(5);
        dragon.addMeshRenderer(new Material());
        gameObjects.add(dragon);


        GameObject donut = new GameObject();
        donut.setMesh(new Donut(1.0f,0.3f,50,20));
        donut.getTransform().posz(6).posy(0.6f);
        donut.addMeshRenderer(new Material(MyGLRenderer.cyan));
        gameObjects.add(donut);

        GameObject cube = new GameObject();
        cube.setMesh(new Cube(1));
        cube.getTransform().posz(6).posx(4);
        cube.addMeshRenderer(new Material(MyGLRenderer.magenta));
        gameObjects.add(cube);

        GameObject pyramid = new GameObject();
        pyramid.setMesh(new Pyramid(80));
        pyramid.getTransform().posx(-4).posz(6);
        pyramid.addMeshRenderer(new Material(MyGLRenderer.yellow));
        gameObjects.add(pyramid);

        GameObject pipe = new GameObject();
        pipe.setMesh(new Pipe(50));
        pipe.getTransform().posz(6).scalex(0.5f).scalez(0.5f);
        pipe.addMeshRenderer(new Material(MyGLRenderer.white));
        gameObjects.add(pipe);

        GameObject cylinder = new GameObject();
        cylinder.setMesh(new Cylinder(50));
        cylinder.getTransform().posz(6).scalez(0.2f).scalex(0.2f);
        cylinder.addMeshRenderer(new Material(MyGLRenderer.blue));
        gameObjects.add(cylinder);

        GameObject tictac = new GameObject();
        tictac.setMesh(new Tictac(50,50));
        tictac.getTransform().posz(6).posx(6).posy(1.7f).scalex(0.7f).scalez(0.7f).scaley(0.8f);
        tictac.addMeshRenderer(new Material(MyGLRenderer.green));
        gameObjects.add(tictac);

        GameObject frustum = new GameObject();
        frustum.setMesh(new Frustum(1.f,0.001f,50));
        frustum.getTransform().posx(6).posz(-6).rotx(45).rotz(45).scaley(2);
        frustum.addMeshRenderer(new Material(MyGLRenderer.magenta));
        gameObjects.add(frustum);

        light = new GameObject();
        light.addComponent(Light.class);
        light.getCompotent(Light.class).setType(LightType.SPOT);
        light.getTransform().rotx(-90.0f).posy(2.4f).posx(-1.5f).posz(1.5f);
        lights.add(light);
        light2 = new GameObject();
        light2.addComponent(Light.class);
        light2.getCompotent(Light.class).setType(LightType.DIRECTIONAL);
        lights.add(light2);
        light3 = new GameObject();
        light3.addComponent(Light.class);
        light3.getCompotent(Light.class).setType(LightType.POINT);
        light3.getTransform().posy(1.f).posz(6.f);
        lights.add(light3);
    }



    /**
     * Init some OpenGL and shaders uniform data to render the simulation scene
     *
     * @param renderer Renderer
     */
    public void initGraphics(MyGLRenderer renderer) {
        MainActivity.log("Initializing graphics");
        // Set the background frame color
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // Allow back face culling !!
        GLES20.glEnable(GLES20.GL_CULL_FACE);
        GLES20.glDepthFunc(GLES20.GL_LESS);
        GLES20.glEnable(GLES20.GL_DEPTH_TEST);
        renderer.getShaders().setNormalizing(true);
        renderer.getShaders().setLighting(true);
        for(GameObject go : gameObjects){
            go.start();
        }
        for (GameObject lgo : lights){
            lgo.start();
        }
        MainActivity.log("Graphics initialized");
    }


    /**
     * Make the scene evoluate, to produce an animation for instance
     * Here, only the viewer rotates
     */
    public void step() {
        this.angley += dx2/10;
        this.anglex += dy2/10;
        if (this.anglex > 70)
            this.anglex = 70;
        else if (this.anglex < -70)
            this.anglex = -70;
        float speedx = dx / 1000;
        float speedy = dy / 1000;
        double yRot = Math.toRadians(this.angley);
        this.posx += speedx * Math.cos(yRot) - speedy * Math.sin(yRot);
        this.posz += speedx * Math.sin(yRot) + speedy * Math.cos(yRot);
    }

    /**
     * Draw the current simulation state
     *
     * @param renderer Renderer
     */
    public void draw(MyGLRenderer renderer) {
        float[] modelviewmatrix = new float[16];

        //MainActivity.log("Starting rendering");

        // Get shader to send uniform data
        MultipleLightingShaders shaders = renderer.getShaders();
        shaders.resetLights();

        // Place viewer in the right position and orientation
        Matrix.setIdentityM(modelviewmatrix, 0);
        // setRotateM instead of rotateM in the next instruction would avoid this initialization...
        Matrix.rotateM(modelviewmatrix, 0, anglex, 1.0F, 0.0F, 0.0F);
        Matrix.rotateM(modelviewmatrix, 0, angley, 0.0F, 1.0F, 0.0F);
        Matrix.translateM(modelviewmatrix, 0, -posx, 0.F, -posz);
        Matrix.translateM(modelviewmatrix, 0, 0.F, -1.6F, 0.F);

        shaders.setViewMatrix(modelviewmatrix);
        for (GameObject light : lights){
            light.update();
        }


        shaders.setModelViewMatrix(modelviewmatrix);
        for(GameObject go : gameObjects){
            go.update();
        }
        for(GameObject go : gameObjects){
            go.lateUpdate();
        }
        //MainActivity.log("Rendering terminated.");
    }
}
