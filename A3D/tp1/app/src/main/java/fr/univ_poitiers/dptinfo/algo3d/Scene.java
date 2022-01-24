package fr.univ_poitiers.dptinfo.algo3d;


import android.opengl.GLES20;
import android.opengl.Matrix;
import android.util.Log;

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

    Room room;
    Room room2;
    Sphere sphere;

    /**
     * An angle used to animate the viewer
     */
    float anglex, angley;

    float posx, posz;
    private Ball ball;
    private Ball ball2;


    /**
     * Constructor : build each wall, the floor and the ceiling as quads
     */
    public Scene() {
        // Init observer's view angles
        angley = 0.F;
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
        room = new Room();
        room2 = new Room();
        //sphere = new Sphere(20,20);
        //sphere = new Sphere(5);
        //sphere.initGraphics();
        ball = new Ball(1.2f,-1.5f,-1.5f,MyGLRenderer.orange);
        ball2 = new Ball(0.3f,1.5f,-1.5f,MyGLRenderer.white);

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


        float[] modelviewmatrixroom = new float[16];

        Matrix.setIdentityM(modelviewmatrixroom, 0);
        Matrix.rotateM(modelviewmatrixroom, 0, 180, 0.0F, 1.0F, 0.0F);
        Matrix.translateM(modelviewmatrixroom, 0, 0.F, 0.F, -6.F);

        Matrix.multiplyMM(modelviewmatrixroom,0,modelviewmatrix, 0, modelviewmatrixroom,0);


        shaders.setModelViewMatrix(modelviewmatrix);
        room.draw(shaders, 1);
        //sphere.draw(shaders);
        ball.draw(shaders,modelviewmatrix);
        ball2.draw(shaders,modelviewmatrix);
        shaders.setModelViewMatrix(modelviewmatrixroom);
        room2.draw(shaders, 2);



        //MainActivity.log("Rendering terminated.");
    }
}
