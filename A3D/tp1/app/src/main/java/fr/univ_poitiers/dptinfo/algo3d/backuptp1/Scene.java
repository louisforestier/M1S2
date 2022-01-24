package fr.univ_poitiers.dptinfo.algo3d.backuptp1;


import android.opengl.GLES20;
import android.opengl.Matrix;

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
    /**
     * 4 quads to represent the walls of the room
     */
    Quad wall1, wall2, wall3, wall4;
    /**
     * A quad to represent a floor
     */
    Quad floor;
    /**
     * A quad to represent the ceiling of the room
     */
    Quad ceiling;

    /**
     * An angle used to animate the viewer
     */
    float anglex, angley;

    float posx, posz;


    /**
     * Constructor : build each wall, the floor and the ceiling as quads
     */
    public Scene() {
        // Init observer's view angles
        angley = 0.F;

        Vec3f point1 = new Vec3f(-3, 0, -3);
        Vec3f point2 = new Vec3f(3, 0, -3);
        Vec3f point3 = new Vec3f(3, 0, 3);
        Vec3f point4 = new Vec3f(-3, 0,3 );
        Vec3f point5 = new Vec3f(-3, 2.5f, -3);
        Vec3f point6 = new Vec3f(3, 2.5f, -3);
        Vec3f point7 = new Vec3f(3, 2.5f, 3);
        Vec3f point8 = new Vec3f(-3, 2.5f, 3);

        // Create the front wall
        this.wall1 = new Quad(point5, point6, point2, point1);

        // Create the right wall
        this.wall2 = new Quad(point6, point7, point3, point2);

        // Create the left wall
        this.wall3 = new Quad(point8, point5, point1, point4);

        // create the back wall
        this.wall4 = new Quad(point7, point8, point4, point3);

        // Create the floor of the room
        this.floor = new Quad(point1, point2, point3, point4);

        // Create the ceiling of the room
        this.ceiling = new Quad(point8, point7, point6, point5);
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
        MainActivity.log("Graphics initialized");
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

        // Draw walls, floor and ceil in selected colors
        // Add code here to draw this.floor, this.ceilling, this.wall<i>
        shaders.setColor(MyGLRenderer.red);
        this.floor.draw(shaders);
        shaders.setColor(MyGLRenderer.blue);
        this.ceiling.draw(shaders);
        shaders.setColor(MyGLRenderer.green);
        this.wall1.draw(shaders);
        shaders.setColor(MyGLRenderer.yellow);
        this.wall2.draw(shaders);
        shaders.setColor(MyGLRenderer.magenta);
        this.wall3.draw(shaders);
        shaders.setColor(MyGLRenderer.orange);
        this.wall4.draw(shaders);
        // Add some wireframe drawings if you want to enhance display
        shaders.setColor(MyGLRenderer.cyan);
        this.floor.drawWireframe(shaders);
        this.ceiling.drawWireframe(shaders);
        this.wall1.drawWireframe(shaders);
        this.wall2.drawWireframe(shaders);
        this.wall3.drawWireframe(shaders);
        this.wall4.drawWireframe(shaders);

        //MainActivity.log("Rendering terminated.");
    }
}
