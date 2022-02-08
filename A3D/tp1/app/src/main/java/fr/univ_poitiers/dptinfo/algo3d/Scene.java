package fr.univ_poitiers.dptinfo.algo3d;


import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

import java.io.InputStream;

/**
 * Class to represent the scene. It includes all the objects to display, in this case a room
 *
 * @author Philippe Meseure
 * @version 1.0
 */
public class Scene {
    /**
     * A constant for the size of the wall
     */
    static final float wallsize = 3.F;
    private Context context;

    private GameObject room;
    private GameObject room2;
    private GameObject armadillo;
    private GameObject ball;
    private GameObject ball2;
    private GameObject donut;
    private GameObject cube;
    private GameObject pyramid;
    private GameObject pipe;
    private GameObject cylinder;
    private GameObject tictac;
    private GameObject plane;
    private GameObject frustum;
    private GameObject room3;

    /**
     * An angle used to animate the viewer
     */
    float anglex, angley;

    float posx, posz;


    /**
     * Constructor : build each wall, the floor and the ceiling as quads
     */
    public Scene(Context current) {
        this.context = current;
        // Init observer's view angles
        angley = 0.F;
        room = new Room(new boolean[]{false, true, true, false},6.F,6.F,2.5F, MyGLRenderer.red, MyGLRenderer.blue, MyGLRenderer.green);
        room2 = new Room(new boolean[]{true, false, false, false},6.F,16.F,2.5F, MyGLRenderer.red, MyGLRenderer.blue, MyGLRenderer.darkgray);
        room2.getTransform().posz(6);
        InputStream stream = context.getResources().openRawResource(R.raw.xyzrgb_dragon);
        ball = new Ball(1.2f,1.5f,1.5f,MyGLRenderer.orange);
        ball2 = new Ball(0.3f,-1.5f,1.5f,MyGLRenderer.white);
        armadillo = new GameObject(MyGLRenderer.lightgray);
        armadillo.setMesh(OBJImporter.importOBJ(stream));
        armadillo.getTransform().posy(1.F).scalex(0.02F).scaley(0.02F).scalez(0.02F).posx(6);
        donut = new GameObject(MyGLRenderer.cyan);
        donut.setMesh(new Donut(1.0f,0.2f,50,20));
        donut.getTransform().posz(6).posy(0.5f);
        cube = new GameObject(MyGLRenderer.magenta);
        cube.setMesh(new Cube(1));
        cube.getTransform().posz(6).posx(4);
        pyramid = new GameObject(MyGLRenderer.yellow);
        pyramid.setMesh(new Pyramid(40));
        pyramid.getTransform().posx(-4).posz(6);
        pipe = new GameObject(MyGLRenderer.white);
        pipe.setMesh(new Pipe(50));
        pipe.getTransform().posz(6);
        cylinder = new GameObject(MyGLRenderer.black);
        cylinder.setMesh(new Cylinder(50));
        cylinder.getTransform().posz(6).scalez(0.5f).scalex(0.5f);
        tictac = new GameObject(MyGLRenderer.green);
        tictac.setMesh(new Tictac(50,50));
        tictac.getTransform().posz(6).posx(6).posy(1.7f).scalex(0.7f).scalez(0.7f).scaley(0.9f);
        plane = new GameObject(MyGLRenderer.orange);
        plane.setMesh(new Plane());
        plane.getTransform().posz(-10);
        frustum = new GameObject(MyGLRenderer.white);
        frustum.setMesh(new Frustum(1.f,0.5f,50));
        frustum.getTransform().posz(-10);
        room3 = new Room(new boolean[]{true, true, true, true},6.f,6.f,2.5f,MyGLRenderer.darkgray,MyGLRenderer.cyan,MyGLRenderer.white);
        room3.getTransform().posx(6);
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
        MainActivity.log("Graphics initialized");
        room.initGraphics();
        room2.initGraphics();
        ball.initGraphics();
        ball2.initGraphics();
        armadillo.initGraphics();
        donut.initGraphics();
        cube.initGraphics();
        pyramid.initGraphics();
        pipe.initGraphics();
        cylinder.initGraphics();
        tictac.initGraphics();
        plane.initGraphics();
        frustum.initGraphics();
        room3.initGraphics();
    }


    /**
     * Make the scene evoluate, to produce an animation for instance
     * Here, only the viewer rotates
     */
    public void step() {
        angley += 0.1F;
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
        NoLightShaders shaders = renderer.getShaders();


        // Place viewer in the right position and orientation
        Matrix.setIdentityM(modelviewmatrix, 0);
        // setRotateM instead of rotateM in the next instruction would avoid this initialization...
        Matrix.rotateM(modelviewmatrix, 0, anglex, 1.0F, 0.0F, 0.0F);
        Matrix.rotateM(modelviewmatrix, 0, angley, 0.0F, 1.0F, 0.0F);
        Matrix.translateM(modelviewmatrix, 0, -posx, 0.F, -posz);
        Matrix.translateM(modelviewmatrix, 0, 0.F, -1.6F, 0.F);



        shaders.setModelViewMatrix(modelviewmatrix);
        room.draw(shaders,modelviewmatrix);
        room2.draw(shaders, modelviewmatrix);
        room3.draw(shaders,modelviewmatrix);
        armadillo.draw(shaders,modelviewmatrix);
        donut.draw(shaders,modelviewmatrix);

        ball.draw(shaders,modelviewmatrix);
        ball2.draw(shaders,modelviewmatrix);
        cube.draw(shaders,modelviewmatrix);
        pyramid.draw(shaders,modelviewmatrix);
        pipe.draw(shaders,modelviewmatrix);
        cylinder.draw(shaders,modelviewmatrix);
        tictac.draw(shaders,modelviewmatrix);
        plane.draw(shaders,modelviewmatrix);
        frustum.draw(shaders,modelviewmatrix);
        //MainActivity.log("Rendering terminated.");
    }
}
