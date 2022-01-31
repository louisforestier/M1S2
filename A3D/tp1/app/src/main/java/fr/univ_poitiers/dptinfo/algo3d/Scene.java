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

    private Room room;
    private Room room2;
    private GameObject armadillo;
    private Ball ball;
    private Ball ball2;

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
        room = new Room(6.F,6.F,2.5F, MyGLRenderer.red, MyGLRenderer.blue, MyGLRenderer.green);
        room2 = new Room(6.F,16.F,2.5F, MyGLRenderer.red, MyGLRenderer.blue, MyGLRenderer.darkgray,0,6,180);
        InputStream stream = context.getResources().openRawResource(R.raw.xyzrgb_dragon);
        ball = new Ball(1.2f,1.5f,1.5f,MyGLRenderer.orange);
        ball2 = new Ball(0.3f,-1.5f,1.5f,MyGLRenderer.white);
        armadillo = new GameObject(MyGLRenderer.lightgray);
        armadillo.setMesh(OBJImporter.importOBJ(stream));
        armadillo.setTransform(new TransformBuilder().posy(1.F).scalex(0.02F).scaley(0.02F).scalez(0.02F).buildTransform());
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
        room.draw(shaders,1);
        room2.draw(shaders, modelviewmatrix,2);

        shaders.setColor(MyGLRenderer.lightgray);
        armadillo.draw(shaders,modelviewmatrix);

        shaders.setModelViewMatrix(modelviewmatrix);
        ball.draw(shaders,modelviewmatrix);
        ball2.draw(shaders,modelviewmatrix);



        //MainActivity.log("Rendering terminated.");
    }
}
