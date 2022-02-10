package fr.univ_poitiers.dptinfo.algo3d;


import static android.view.MotionEvent.INVALID_POINTER_ID;

import static androidx.core.view.MotionEventCompat.getActionMasked;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.view.MotionEvent;

import androidx.core.view.MotionEventCompat;

/**
 * Class to described the surface view. Mainly based on well-known code.
 */
public class MyGLSurfaceView extends GLSurfaceView {
    private final MyGLRenderer renderer;
    private final Scene scene;
    private float mLastTouchX;
    private float mLastTouchY;
    private float mPosX;
    private float mPosY;

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

    private int mActivePointerId = INVALID_POINTER_ID;

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

            switch (e.getActifloaton()) {
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
        //TODO: https://android-developers.googleblog.com/2010/06/making-sense-of-multitouch.html
        int screenWidth = getWidth();

/*
        switch (e.getAction()) {
            case MotionEvent.ACTION_DOWN:
                Log.d("JOYSTICK", "action down");
                Log.d("JOYSTICK", "action point index mask :" + MotionEvent.ACTION_POINTER_INDEX_MASK);
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
                Log.d("JOYSTICK", "action pointer down");
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
            case MotionEvent.ACTION_UP:
                Log.d("JOYSTICK", "action up");
                break;
            case MotionEvent.ACTION_POINTER_UP:
                Log.d("JOYSTICK", "action pointer up");
                break;
        }
*/

/*
        switch (e.getAction() & MotionEvent.ACTION_MASK) {
            case MotionEvent.ACTION_DOWN:
            {
                Log.d("Controlls", "Action Down "+ pointerId);
                Log.d("Controlls", "Coordinates "+ e.getX() + " "+ e.getY());
                break;
            }

            case MotionEvent.ACTION_UP:
            {
                Log.d("Controlls", "Action UP "+ pointerId);
                Log.d("Controlls", "Coordinates "+ e.getX() + " "+ e.getY());
                break;
            }

            case MotionEvent.ACTION_POINTER_DOWN:
            {
                Log.d("Controlls", "Action Pointer Down "+ pointerId);
                Log.d("Controlls", "Coordinates "+ e.getX() + " "+ e.getY());
                break;
            }

            case MotionEvent.ACTION_POINTER_UP:
            {
                index = (e.getAction() & MotionEvent.ACTION_POINTER_INDEX_MASK) >> MotionEvent.ACTION_POINTER_INDEX_SHIFT;
                int pointId = e.getPointerId(index);
                Log.d("Controlls", "Action Pointer UP "+ pointId);
                Log.d("Controlls", "Coordinates "+ e.getX(index) + " "+  e.getY(index));
                break;
            }
            case MotionEvent.ACTION_MOVE:
                index = (e.getAction() & MotionEvent.ACTION_POINTER_INDEX_MASK) >> MotionEvent.ACTION_POINTER_INDEX_SHIFT;
                int pointId = e.getPointerId(index);
                Log.d("Controlls", "Action Move "+ pointId);
                Log.d("Controlls", "Action Move index"+ index);
                Log.d("Controlls", "Coordinates "+ e.getX(index) + " "+  e.getY(index));
        }
*/


        int action = e.getActionMasked();
// Get the index of the pointer associated with the action.
        int index = e.getActionIndex();
        int xPos = -1;
        int yPos = -1;
        int pointerIndex;
        switch (action) {
            case MotionEvent.ACTION_DOWN:
                pointerIndex = MotionEventCompat.getActionIndex(e);
                x = MotionEventCompat.getX(e, pointerIndex);
                y = MotionEventCompat.getY(e, pointerIndex);

                // Remember where we started (for dragging)
                mLastTouchX = x;
                mLastTouchY = y;
                // Save the ID of this pointer (for dragging)
                mActivePointerId = MotionEventCompat.getPointerId(e, 0);
                break;
            case MotionEvent.ACTION_MOVE: {
                // Find the index of the active pointer and fetch its position
                pointerIndex = MotionEventCompat.findPointerIndex(e, mActivePointerId);

                x = MotionEventCompat.getX(e, pointerIndex);
                y = MotionEventCompat.getY(e, pointerIndex);

                // Calculate the distance moved
                final float dx = x - mLastTouchX;
                final float dy = y - mLastTouchY;

                mPosX += dx;
                mPosY += dy;

                invalidate();

                // Remember this touch position for the next move event
                mLastTouchX = x;
                mLastTouchY = y;

                break;
            }
            case MotionEvent.ACTION_UP: {
                mActivePointerId = INVALID_POINTER_ID;
                break;
            }

            case MotionEvent.ACTION_CANCEL: {
                mActivePointerId = INVALID_POINTER_ID;
                break;
            }
            case MotionEvent.ACTION_POINTER_UP: {

                pointerIndex = MotionEventCompat.getActionIndex(e);
                final int pointerId = MotionEventCompat.getPointerId(e, pointerIndex);

                if (pointerId == mActivePointerId) {
                    // This was our active pointer going up. Choose a new
                    // active pointer and adjust accordingly.
                    final int newPointerIndex = pointerIndex == 0 ? 1 : 0;
                    mLastTouchX = MotionEventCompat.getX(e, newPointerIndex);
                    mLastTouchY = MotionEventCompat.getY(e, newPointerIndex);
                    mActivePointerId = MotionEventCompat.getPointerId(e, newPointerIndex);
                }
                break;
            }

        }
        Log.d("DEBUG_TAG","The action is " + actionToString(action));

        if (e.getPointerCount() > 1) {
            Log.d("DEBUG_TAG","Multitouch event");
            // The coordinates of the current screen contact, relative to
            // the responding View or Activity.
            Log.d("DEBUG_TAG", "Coordinates "+ e.getX() + " "+ e.getY());
            xPos = (int)e.getX(index);
            yPos = (int)e.getY(index);

        } else {
            // Single touch event
            Log.d("DEBUG_TAG","Single touch event");
            Log.d("DEBUG_TAG", "Coordinates "+ e.getX() + " "+ e.getY());

            xPos = (int)e.getX(index);
            yPos = (int)e.getY(index);
        }

        /**
         * Controle avec la moitie  gauche de l'écran qui régit le déplacement et la moitié droite qui régit la rotation.
         *//*
        switch (e.getAction()) {
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
        }*/

        previousx = x;
        previousy = y;
        this.requestRender();
        return true;
    }

    public static String actionToString(int action) {
        switch (action) {

            case MotionEvent.ACTION_DOWN: return "Down";
            case MotionEvent.ACTION_MOVE: return "Move";
            case MotionEvent.ACTION_POINTER_DOWN: return "Pointer Down";
            case MotionEvent.ACTION_UP: return "Up";
            case MotionEvent.ACTION_POINTER_UP: return "Pointer Up";
            case MotionEvent.ACTION_OUTSIDE: return "Outside";
            case MotionEvent.ACTION_CANCEL: return "Cancel";
        }
        return "";
    }

}
