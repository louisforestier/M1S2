package fr.univ_poitiers.dptinfo.algo3d;


import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.MotionEvent;

/**
 * Class to described the surface view. Mainly based on well-known code.
 */
public class MyGLSurfaceView extends GLSurfaceView {
    private final MyGLRenderer renderer;
    private final Scene scene;
    public MyGLSurfaceView(Context context, Scene scene) {
        super(context);
        this.scene = scene;

        // Create an OpenGL ES 2.0 context.
        setEGLContextClientVersion(2);

        // Set the Renderer for drawing on the GLSurfaceView
        this.renderer = new MyGLRenderer(this, scene);
        setRenderer(this.renderer);

        // Render the view only when there is a change in the drawing data
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        //setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);
    }



    private final float SCALE_FACTOR = 0.005F;
    private float previousx;
    private float previousy;
    private float previousx2;
    private float previousy2;
    private float leftJoystickOriginX;
    private float leftJoystickOriginY;
    private float rightJoystickOriginX;
    private float rightJoystickOriginY;

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        // MotionEvent reports input details from the touch screen
        // and other input controls. In this case, you are only
        // interested in events where the touch position changed.
        // MainActivity.log("Event");

        float x = e.getX();
        float y = e.getY();

        float deltax = x - previousx; // motion along x axis
        float deltay = y - previousy; // motion along y axis
        /** Controle avec un doigt pour la rotation et deux doigts pour le déplacement*/
        /*
        if (e.getPointerCount() == 2) {
            float x2;
            float y2;
            x2 = e.getX(1);
            y2 = e.getY(1);
            float deltax2 = x2 - previousx2;
            float deltay2 = y2 - previousy2;
            MainActivity.log("Sin :"+Math.sin(scene.angley * Math.PI/180));
            MainActivity.log("Cos :"+Math.cos(scene.angley * Math.PI/180));

            switch (e.getAction()) {
                case MotionEvent.ACTION_MOVE:
                    if (deltax * deltax2 > 0 || deltay * deltay2 > 0) {
                        MainActivity.log("dx :"+(deltax + deltax2) / (2 * 100));
                        MainActivity.log("dz :"+(deltay + deltay2) / (2 * 100));
                        MainActivity.log("dx * sin:"+(deltax + deltax2) / (2 * 100) * Math.sin(scene.angley * Math.PI/180));
                        MainActivity.log("dz * cos:"+(deltay + deltay2) / (2 * 100) * Math.cos(scene.angley * Math.PI/180));
                        float speedx = (deltax + deltax2) / (2 * 100);
                        float speedy = (deltay + deltay2) / (2 * 100);
                        double yRot = Math.toRadians(scene.angley);
                        scene.posx += speedx * Math.cos(yRot) - speedy*Math.sin(yRot);
                        scene.posz += speedx * Math.sin(yRot) + speedy*Math.cos(yRot);
                    }
                    break;
            }
            previousx2 = x2;
            previousy2 = y2;
        } else if (e.getPointerCount() == 1) {
            switch (e.getAction()) {
                case MotionEvent.ACTION_MOVE:
                    // to complete
                    // You can use deltax and deltay to make mouse motion control the position
                    // and/or the orientation of the viewer
                    scene.angley += deltax;
                    scene.anglex += deltay;
                    break;
            }
        }*/
        /**
         * Controle avec la moitie  gauche de l'écran qui régit le déplacement et la moitié droite qui régit la rotation.
         */
        //TODO: https://android-developers.googleblog.com/2010/06/making-sense-of-multitouch.html
        int screenWidth = getWidth();
        switch (e.getAction()) {
            case MotionEvent.ACTION_DOWN:
                if (x > screenWidth /2) {
                    rightJoystickOriginX = e.getX();
                    rightJoystickOriginY = e.getY();
                    Log.d("JOYSTICK", "right joystick");
                }
                if (x <= screenWidth /2){
                    leftJoystickOriginX = x;
                    leftJoystickOriginY = y;
                    Log.d("JOYSTICK", "left joystick");
                }
                break;
            case MotionEvent.ACTION_POINTER_DOWN:
                if (e.getPointerCount() == 2) {
                    if (x > screenWidth / 2) {
                        rightJoystickOriginX = e.getX();
                        rightJoystickOriginY = e.getY();
                        Log.d("JOYSTICK", "right joystick");
                    }
                    if (x <= screenWidth / 2) {
                        leftJoystickOriginX = x;
                        leftJoystickOriginY = y;
                        Log.d("JOYSTICK", "left joystick");
                    }
                }
                break;
            case MotionEvent.ACTION_MOVE:
                if (e.getX() > screenWidth / 2 && previousx > screenWidth /2) {
                    scene.angley += deltax;
                    scene.anglex += deltay;
                    if (scene.anglex > 70)
                        scene.anglex = 70;
                    else if (scene.anglex < -70)
                        scene.anglex = -70;
                }
                if (e.getX() <= screenWidth / 2 && previousx <= screenWidth/2) {
                    float speedx = deltax / 100;
                    float speedy = deltay / 100;
                    double yRot = Math.toRadians(scene.angley);
                    scene.posx += speedx * Math.cos(yRot) - speedy * Math.sin(yRot);
                    scene.posz += speedx * Math.sin(yRot) + speedy * Math.cos(yRot);
                }

                if (e.getPointerCount() == 2) {
                    float x2 = e.getX(1);
                    float y2 = e.getY(1);

                    float deltax2 = x2 - previousx2; // motion along x axis
                    float deltay2 = y2 - previousy2; // motion along y axis

                    if (e.getX() > screenWidth / 2) {
                        scene.angley += deltax;
                        scene.anglex += deltay;
                        if (scene.anglex > 70)
                            scene.anglex = 70;
                        else if (scene.anglex < -70)
                            scene.anglex = -70;
                    } else if (e.getX(1) > screenWidth / 2) {
                        scene.angley += deltax2;
                        scene.anglex += deltay2;
                        if (scene.anglex > 70)
                            scene.anglex = 70;
                        else if (scene.anglex < -70)
                            scene.anglex = -70;
                    }
                    if (e.getX() <= screenWidth / 2) {
                        float speedx = deltax / 100;
                        float speedy = deltay / 100;
                        double yRot = Math.toRadians(scene.angley);
                        scene.posx += speedx * Math.cos(yRot) - speedy * Math.sin(yRot);
                        scene.posz += speedx * Math.sin(yRot) + speedy * Math.cos(yRot);
                    } else if (e.getX(1) <= screenWidth / 2) {
                        float speedx = deltax2 / 100;
                        float speedy = deltay2 / 100;
                        double yRot = Math.toRadians(scene.angley);
                        scene.posx += speedx * Math.cos(yRot) - speedy * Math.sin(yRot);
                        scene.posz += speedx * Math.sin(yRot) + speedy * Math.cos(yRot);

                    }
                    previousx2 = x2;
                    previousy2 = y2;


                }
                break;
        }

        previousx = x;
        previousy = y;
        this.requestRender();
        return true;
    }

}
