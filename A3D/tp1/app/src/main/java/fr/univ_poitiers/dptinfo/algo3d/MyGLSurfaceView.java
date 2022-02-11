package fr.univ_poitiers.dptinfo.algo3d;


import static android.view.MotionEvent.INVALID_POINTER_ID;

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
    private float leftJoystickOriginX;
    private float leftJoystickOriginY;
    private float rightJoystickOriginX;
    private float rightJoystickOriginY;
    private int leftJoystickId=-1;
    private int rightJoystickId=-1;


    private int activePointerId = INVALID_POINTER_ID;

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        // MotionEvent reports input details from the touch screen
        // and other input controls. In this case, you are only
        // interested in events where the touch position changed.
        // MainActivity.log("Event");
        if(e.getPointerCount() > 2){
            if (leftJoystickId != -1) {
                leftJoystickOriginX = e.getX(leftJoystickId);
                leftJoystickOriginY = e.getY(leftJoystickId);
            }
            if (rightJoystickId != -1) {
                rightJoystickOriginX = e.getX(rightJoystickId);
                rightJoystickOriginY = e.getY(rightJoystickId);
            }
            scene.dx = 0;
            scene.dy = 0;
            scene.dx2 = 0;
            scene.dy2 = 0;
            scene.anglex = 0;
            scene.angley = 0;
            scene.posx = 0;
            scene.posz = 0;
        } else {

            int screenWidth = getWidth();

            float x;
            float y;

            int action = e.getActionMasked();
// Get the index of the pointer associated with the action.
            int pointerIndex;
            switch (action) {
                case MotionEvent.ACTION_DOWN:
                    pointerIndex = e.getActionIndex();
                    x = e.getX(pointerIndex);
                    y = e.getY(pointerIndex);

                    if (x < screenWidth / 2) {
                        leftJoystickOriginX = x;
                        leftJoystickOriginY = y;
                        leftJoystickId = e.getPointerId(0);
//                    Log.d("JOYSTICKS", "left joystick " + pointerIndex);
                    } else {
                        rightJoystickOriginX = x;
                        rightJoystickOriginY = y;
                        rightJoystickId = e.getPointerId(0);
//                    Log.d("JOYSTICKS", "right joystick " + pointerIndex);
                    }
                    // Save the ID of this pointer (for dragging)
                    activePointerId = e.getPointerId(0);
//                Log.d("JOYSTICKS", "Action DOWN "+ pointerIndex);
//                Log.d("JOYSTICKS", "Coordinates "+ e.getX(index) + " "+  e.getY(index));

                    break;
                case MotionEvent.ACTION_POINTER_DOWN: {
                    pointerIndex = e.getActionIndex();
                    x = e.getX(pointerIndex);
                    y = e.getY(pointerIndex);
                    if (x < screenWidth / 2 && leftJoystickId == -1) {
                        leftJoystickOriginX = x;
                        leftJoystickOriginY = y;
                        leftJoystickId = pointerIndex;
//                    Log.d("JOYSTICKS", "left joystick "+pointerIndex);
                    } else if (x >= screenWidth / 2 && rightJoystickId == -1) {
                        rightJoystickOriginX = x;
                        rightJoystickOriginY = y;
                        rightJoystickId = pointerIndex;
//                    Log.d("JOYSTICKS", "right joystick " +pointerIndex);
                    }
//                Log.d("JOYSTICKS", "Action pointer DOWN "+ pointerIndex);
//                Log.d("JOYSTICKS", "Coordinates "+ x + " "+ y);
                    break;
                }
                case MotionEvent.ACTION_MOVE: {
                    int pointerCount = e.getPointerCount();
                    for (int i = 0; i < pointerCount; ++i) {
                        pointerIndex = i;
                        x = e.getX(pointerIndex);
                        y = e.getY(pointerIndex);
                        int pointerId = e.getPointerId(pointerIndex);
                        if (pointerId == leftJoystickId) {
                            scene.dx = (x - leftJoystickOriginX) / 2;
                            scene.dy = (y - leftJoystickOriginY) / 2;
                        } else if (pointerId == rightJoystickId) {
                            scene.dx2 = (x - rightJoystickOriginX) / 8;
                            scene.dy2 = (y - rightJoystickOriginY) / 8;

                        }
                    }
                    break;
                }
                case MotionEvent.ACTION_UP: {
                    activePointerId = INVALID_POINTER_ID;
                    leftJoystickId = -1;
                    rightJoystickId = -1;
                    scene.dx = 0;
                    scene.dy = 0;
                    scene.dx2 = 0;
                    scene.dy2 = 0;
                    break;
                }

                case MotionEvent.ACTION_POINTER_UP: {

                    pointerIndex = e.getActionIndex();
                    final int pointerId = e.getPointerId(pointerIndex);
                    if (pointerId == leftJoystickId) {
                        leftJoystickId = -1;
                        scene.dx = 0;
                        scene.dy = 0;
                    } else if (pointerId == rightJoystickId) {
                        rightJoystickId = -1;
                        scene.dx2 = 0;
                        scene.dy2 = 0;
                    }
                    if (pointerId == activePointerId) {
                        final int newPointerIndex = pointerIndex == 0 ? 1 : 0;
                        if (pointerId == leftJoystickId)
                            leftJoystickId = newPointerIndex;
                        else rightJoystickId = newPointerIndex;
                        activePointerId = e.getPointerId(newPointerIndex);
//                    Log.d("JOYSTICKS", "Action Pointer UP new Pointer Index"+ newPointerIndex);
                    }
//                Log.d("JOYSTICKS", "Action Pointer UP id"+ pointerId);
//                Log.d("JOYSTICKS", "Action Pointer UP index"+ pointerIndex);
                    break;
                }
            }
        }
//        Log.d("JOYSTICKS", "dx = " + scene.dx);
//        Log.d("JOYSTICKS", "dy = " + scene.dy);
//        Log.d("JOYSTICKS", "dx2 = " + scene.dx2);
//        Log.d("JOYSTICKS", "dy2 = " + scene.dy2);
        this.requestRender();
        return true;
    }
}
