package fr.univ_poitiers.dptinfo.algo3d;


import android.content.Context;
import android.opengl.GLES20;
import android.opengl.Matrix;

import java.io.InputStream;

import fr.univ_poitiers.dptinfo.algo3d.gameobject.Ball;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.GameObject;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.LightGameObject;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.Room;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Cube;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Cylinder;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Donut;
import fr.univ_poitiers.dptinfo.algo3d.mesh.DrawMode;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Frustum;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Pipe;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Pyramid;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Tictac;
import fr.univ_poitiers.dptinfo.algo3d.objimporter.OBJImporter;
import fr.univ_poitiers.dptinfo.algo3d.shaders.LightingShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShadingMode;

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
    private Context context;

    private GameObject room;
    private GameObject room2;
    private GameObject armadillo;
    private GameObject armadillo2;
    private GameObject dragon;
    private GameObject ball;
    private GameObject ball2;
    private GameObject donut;
    private GameObject cube;
    private GameObject pyramid;
    private GameObject pipe;
    private GameObject cylinder;
    private GameObject tictac;
    private GameObject frustum;
    private GameObject room3;
    private GameObject room4;

    private LightGameObject light;
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
        this.context = current;
        // Init observer's view angles
        angley = 0.F;
        room = new Room(new boolean[]{false, false, true, true},6.F,6.F,2.5F, MyGLRenderer.green, MyGLRenderer.darkgray, MyGLRenderer.lightgray);
        room2 = new Room(new boolean[]{true, false, false, false},6.F,16.F,2.5F, MyGLRenderer.red, MyGLRenderer.darkgray, MyGLRenderer.lightgray);
        room2.getTransform().posz(6);
        room3 = new Room(new boolean[]{true, true, true, true},6.f,6.f,2.5f,MyGLRenderer.cyan,MyGLRenderer.darkgray,MyGLRenderer.lightgray);
        room3.getTransform().posx(6);
        room4 = new Room(new boolean[]{false,true,false,true},6.f,6.f,4.5f,MyGLRenderer.orange,MyGLRenderer.darkgray,MyGLRenderer.lightgray);
        //je pourrais aussi créer mes portes sur les autres murs mais c'est pour vérifier que la rotation fonctionne correctement
        room4.getTransform().posx(6).posz(-6).roty(90).rotx(180).posy(2.f);
        ball = new Ball(1.2f,1.5f,1.5f,MyGLRenderer.orange);
        ball2 = new Ball(0.3f,-1.5f,1.5f,MyGLRenderer.gray);
        armadillo = new GameObject(MyGLRenderer.lightgray);
        InputStream stream = context.getResources().openRawResource(R.raw.armadillo);
        armadillo.setMesh(OBJImporter.importOBJ(stream, ShadingMode.SMOOTH_SHADING));
        armadillo.getTransform().posy(1.F).scalex(0.01F).scaley(0.01F).scalez(0.01F).posx(7.5f);
        armadillo2 = new GameObject(MyGLRenderer.lightgray);
        stream = context.getResources().openRawResource(R.raw.armadillo_with_normals);
        armadillo2.setMesh(OBJImporter.importOBJ(stream,ShadingMode.SMOOTH_SHADING));
        armadillo2.getTransform().posy(1.F).scalex(0.01F).scaley(0.01F).scalez(0.01F).posx(7.5f).posz(1.f);
        stream = context.getResources().openRawResource(R.raw.xyzrgb_dragon);
        dragon = new GameObject(MyGLRenderer.white);
        dragon.setMesh(OBJImporter.importOBJ(stream, ShadingMode.FLAT_SHADING));
        dragon.getTransform().posy(1.f).scalex(0.02f).scaley(0.02f).scalez(0.02f).posx(5);
        donut = new GameObject(MyGLRenderer.cyan);
        donut.setMesh(new Donut(1.0f,0.3f,50,20));
        donut.getTransform().posz(6).posy(0.6f);
        cube = new GameObject(MyGLRenderer.magenta);
        cube.setMesh(new Cube(1));
        cube.getTransform().posz(6).posx(4);
        pyramid = new GameObject(MyGLRenderer.yellow);
        pyramid.setMesh(new Pyramid(80));
        pyramid.getTransform().posx(-4).posz(6);
        pipe = new GameObject(MyGLRenderer.white);
        pipe.setMesh(new Pipe(50));
        pipe.getTransform().posz(6).scalex(0.5f).scalez(0.5f);
        cylinder = new GameObject(MyGLRenderer.blue);
        cylinder.setMesh(new Cylinder(50));
        cylinder.getTransform().posz(6).scalez(0.2f).scalex(0.2f);
        tictac = new GameObject(MyGLRenderer.green);
        tictac.setMesh(new Tictac(50,50));
        tictac.getTransform().posz(6).posx(6).posy(1.7f).scalex(0.7f).scalez(0.7f).scaley(0.8f);
        frustum = new GameObject(MyGLRenderer.magenta);
        frustum.setMesh(new Frustum(1.f,0.001f,50));
        frustum.getTransform().posx(6).posz(-6).rotx(45).rotz(45).scaley(2);

        light = new LightGameObject(new Light(new float[]{0.2f,0.2f,0.2f,1.f},new float[]{0.8f,0.8f,0.8f,1.f},new float[]{0.8f,0.8f,0.8f,1.f},1.f,0.09f,0.032f));
        light.getTransform().posy(1.6f).posz(6.f);
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
        renderer.getShaders().setNormalizing(true);
        renderer.getShaders().setLighting(true);
        room.initGraphics();
        room2.initGraphics();
        room3.initGraphics();
        room4.initGraphics();
        ball.initGraphics();
        ball2.initGraphics();
        armadillo.initGraphics();
        armadillo2.initGraphics();
        dragon.initGraphics();
        donut.initGraphics();
        cube.initGraphics();
        pyramid.initGraphics();
        pipe.initGraphics();
        cylinder.initGraphics();
        tictac.initGraphics();
        frustum.initGraphics();
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
        LightingShaders shaders = renderer.getShaders();



        // Place viewer in the right position and orientation
        Matrix.setIdentityM(modelviewmatrix, 0);
        // setRotateM instead of rotateM in the next instruction would avoid this initialization...
        Matrix.rotateM(modelviewmatrix, 0, anglex, 1.0F, 0.0F, 0.0F);
        Matrix.rotateM(modelviewmatrix, 0, angley, 0.0F, 1.0F, 0.0F);
        Matrix.translateM(modelviewmatrix, 0, -posx, 0.F, -posz);
        Matrix.translateM(modelviewmatrix, 0, 0.F, -1.6F, 0.F);

        renderer.getShaders().setLightPosition(light.getPos(modelviewmatrix));
        renderer.getShaders().setLightColor(light.getLight().getDiffuse());
        renderer.getShaders().setAmbiantLight(light.getLight().getAmbient());
        renderer.getShaders().setLightSpecular(light.getLight().getSpecular());
        renderer.getShaders().setLightAttenuation(light.getLight().getConstant(),light.getLight().getLinear(), light.getLight().getQuadratic());

        shaders.setModelViewMatrix(modelviewmatrix);

        room.draw(shaders,modelviewmatrix);
        room2.draw(shaders, modelviewmatrix);
        room3.draw(shaders,modelviewmatrix);
        room4.draw(shaders,modelviewmatrix);

        ball.draw(shaders,modelviewmatrix);
        ball2.draw(shaders,modelviewmatrix);
        armadillo.draw(shaders,modelviewmatrix);
        armadillo2.draw(shaders,modelviewmatrix);
        dragon.draw(shaders, modelviewmatrix);

        cube.draw(shaders,modelviewmatrix, DrawMode.TRIANGLES_AND_WIREFRAME);
        donut.draw(shaders,modelviewmatrix);
        pyramid.draw(shaders,modelviewmatrix);
        pipe.draw(shaders,modelviewmatrix);
        cylinder.draw(shaders,modelviewmatrix);
        tictac.draw(shaders,modelviewmatrix);
        frustum.draw(shaders,modelviewmatrix);



        //MainActivity.log("Rendering terminated.");
    }
}
