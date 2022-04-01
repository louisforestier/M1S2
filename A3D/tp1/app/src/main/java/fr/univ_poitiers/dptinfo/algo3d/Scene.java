package fr.univ_poitiers.dptinfo.algo3d;


import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

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
import fr.univ_poitiers.dptinfo.algo3d.mesh.Plane;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Pyramid;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Tictac;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShaderManager;

/**
 * Class to represent the scene. It includes all the objects to display, in this case a room
 *
 * @author Philippe Meseure
 * @version 1.0
 */
public class Scene {
    private GameObject mirror;
    private GameObject light;
    public final GameObject light2;
    private final GameObject light3;

    private List<GameObject> gameObjects = new ArrayList<>();

    /**
     * An angle used to animate the viewer
     */
    float anglex, angley;

    float posx, posz;
    float dx,dy,dx2,dy2;
    private final Material wallMaterial;
    private final Material ceilingMaterial;
    private final Material floorMaterial;
    private final Material floorMaterial2;
    private final Material sunMaterial;
    private final Material earthMaterial;
    private float[] modelviewmatrix = new float[16];
    public final GameObject cube2;


    /**
     * Constructor : build each wall, the floor and the ceiling as quads
     */
    public Scene(Context current) {
        /**
         * A constant for the size of the wall
         */
        // Init observer's view angles
        angley = 0.F;
        ceilingMaterial = new Material(MyGLRenderer.darkgray);
        wallMaterial = new Material(MyGLRenderer.lightgray);
        floorMaterial = new Material(new float[]{1.f,1.f,1.f,0.5f});
        floorMaterial2 = new Material(new float[]{1.f,1.f,1.f,0.5f});
        sunMaterial = new Material(MyGLRenderer.orange);
        earthMaterial = new Material(MyGLRenderer.white);
        GameObject room = new Room(new boolean[]{false, false, true, true}, 6.F, 6.F, 2.5F, floorMaterial, ceilingMaterial, wallMaterial);
        gameObjects.add(room);
        GameObject room2 = new Room(new boolean[]{true, false, false, false}, 6.F, 16.F, 2.5F, floorMaterial2, ceilingMaterial, wallMaterial);
        room2.getTransform().posz(6);
        gameObjects.add(room2);
        GameObject room3 = new Room(new boolean[]{true, true, true, true}, 6.f, 6.f, 2.5f, floorMaterial, ceilingMaterial, wallMaterial);
        room3.getTransform().posx(6);
        gameObjects.add(room3);
        GameObject room4 = new Room(new boolean[]{false, true, false, true}, 6.f, 6.f, 4.5f, floorMaterial2, ceilingMaterial, wallMaterial);
        room4.getTransform().posx(6).posz(-6).roty(90);
        gameObjects.add(room4);

        mirror = new GameObject();
        mirror.setMesh(Plane.INSTANCE);
        mirror.getTransform().posx(6).posz(-6).scalex(0.4f).scalez(0.4f).posy(0.01f);
        mirror.addMeshRenderer(new Material(new float[]{0.f,0.f,1.f,0.4f}));
        mirror.addComponent(Mirror.class);
        gameObjects.add(mirror);

        GameObject ball = new Ball(1.2f, 1.5f, 1.5f, sunMaterial);
        gameObjects.add(ball);

        GameObject ball2 = new Ball(0.3f, -1.5f, 1.5f, earthMaterial);
        ball2.getTransform().posy(1.8f);
        gameObjects.add(ball2);


/*
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
*/


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

        cube2 = new GameObject();
        cube2.setMesh(new Cube(1));
        cube2.getTransform().posz(-1.5f).posx(-0.5f);
        cube2.addMeshRenderer(new Material(MyGLRenderer.magenta));
        gameObjects.add(cube2);

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
        gameObjects.add(light);
        light2 = new GameObject();
        light2.addComponent(Light.class);
        light2.getCompotent(Light.class).setType(LightType.DIRECTIONAL);
        light2.getTransform().posy(10).roty(-30).rotx(-50);
        gameObjects.add(light2);
        light3 = new GameObject();
        light3.addComponent(Light.class);
        light3.getCompotent(Light.class).setType(LightType.POINT);
        light3.getTransform().posy(1.f).posz(6.f);
        gameObjects.add(light3);
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
        wallMaterial.setTextureId(MyGLRenderer.loadTexture(renderer.getView().getContext(),R.drawable.wall));
        ceilingMaterial.setTextureId(MyGLRenderer.loadTexture(renderer.getView().getContext(),R.drawable.ceiling));
        floorMaterial.setTextureId(MyGLRenderer.loadTexture(renderer.getView().getContext(),R.drawable.tiles1));
        floorMaterial2.setTextureId(MyGLRenderer.loadTexture(renderer.getView().getContext(),R.drawable.tiles2));
        sunMaterial.setTextureId(MyGLRenderer.loadTexture(renderer.getView().getContext(),R.drawable.sun));
        earthMaterial.setTextureId(MyGLRenderer.loadTexture(renderer.getView().getContext(),R.drawable.earth));
        for(GameObject go : gameObjects){
            go.start();
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

        if (ShaderManager.isRender()) {
            shaders.setViewMatrix(modelviewmatrix);
            shaders.setModelViewMatrix(modelviewmatrix);
        }else{
            renderer.getShadowShader().setModelViewMatrix(modelviewmatrix);
        }
        for (GameObject go : gameObjects){
            go.earlyUpdate();
        }


        for(GameObject go : gameObjects){
            go.update();
        }
        for(GameObject go : gameObjects){
            go.lateUpdate();
        }
        //MainActivity.log("Rendering terminated.");
    }

    public void setUpMatrix(MyGLRenderer renderer){
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
        shaders.setModelViewMatrix(modelviewmatrix);
    }
    public void setUpReflexionMatrix(MyGLRenderer renderer){
        MultipleLightingShaders shaders = renderer.getShaders();
        shaders.resetLights();

        // Place viewer in the right position and orientation
        Matrix.setIdentityM(modelviewmatrix, 0);
        // setRotateM instead of rotateM in the next instruction would avoid this initialization...
        Matrix.rotateM(modelviewmatrix, 0, anglex, 1.0F, 0.0F, 0.0F);
        Matrix.rotateM(modelviewmatrix, 0, angley, 0.0F, 1.0F, 0.0F);
        Matrix.translateM(modelviewmatrix, 0, -posx, 0.F, -posz);
        Matrix.translateM(modelviewmatrix, 0, 0.F, -1.6F, 0.F);
        Matrix.multiplyMM(modelviewmatrix,0,modelviewmatrix,0,getReflexionMatrix(mirror),0);
        shaders.setViewMatrix(modelviewmatrix);
        shaders.setModelViewMatrix(modelviewmatrix);
    }

    public float[] getReflexionMatrix(GameObject mirror) {
        Vec3f point = new Vec3f(mirror.getTransform().getPosx(),mirror.getTransform().getPosy(),mirror.getTransform().getPosz());
        Vec3f rotation = new Vec3f(mirror.getTransform().getRotx(),mirror.getTransform().getRoty(),mirror.getTransform().getRotz());
        float[] rotationMatrix = new float[16];
        Matrix.setRotateM(rotationMatrix,0,rotation.z,0f,0f,1f);
        Matrix.rotateM(rotationMatrix,0,rotation.x,1f,0f,0f);
        Matrix.rotateM(rotationMatrix,0,rotation.y,0f,1f,0f);
        float[] rawNormal = new float[]{0,1,0,1};
        Matrix.multiplyMV(rawNormal,0,rotationMatrix,0,rawNormal,0);
        Vec3f  normal = new Vec3f(rawNormal[0],rawNormal[1],rawNormal[2]);
        float[] matrix = new float[]{
                1-2*normal.x*normal.x,
                -2*normal.x* normal.y,
                -2*normal.x* normal.z,
                0,
                -2*normal.x* normal.y,
                1-2*normal.y*normal.y,
                -2*normal.y* normal.z,
                0,
                -2*normal.x* normal.z,
                -2*normal.y* normal.z,
                1-2*normal.z*normal.z,
                0,
                2*(point.dotProduct(normal))*normal.x,
                2*(point.dotProduct(normal))*normal.y,
                2*(point.dotProduct(normal))*normal.z,
                1
        };
        return matrix;
    }


    public void earlyUpdate(){
        for (GameObject go : gameObjects){
            go.earlyUpdate();
        }
    }
    public void update(){
        for(GameObject go : gameObjects){
            go.update();
        }
    }

    public void lateUpdate(){
        for(GameObject go : gameObjects){
            if (go == mirror) continue;
            go.lateUpdate();
        }
    }

    public void finalRendering(){
        for (GameObject go : gameObjects){
            if (go == mirror){
                GLES20.glEnable(GLES20.GL_BLEND);
                GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA,GLES20.GL_ONE_MINUS_SRC_ALPHA);
                go.lateUpdate();
                GLES20.glDisable(GLES20.GL_BLEND);
            }
            else go.lateUpdate();
        }
    }

    public void fillStencil(){
        mirror.lateUpdate();
    }



}
