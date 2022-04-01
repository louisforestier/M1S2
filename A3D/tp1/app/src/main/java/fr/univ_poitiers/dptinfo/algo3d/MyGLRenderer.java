package fr.univ_poitiers.dptinfo.algo3d;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;
import android.opengl.Matrix;

import fr.univ_poitiers.dptinfo.algo3d.shaders.Light;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShaderManager;
import fr.univ_poitiers.dptinfo.algo3d.shaders.DepthShader;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShadowShaders;


/**
 * Class to represent the rendering of the scene
 * @author Philippe Meseure
 * @version 1.0
 */
public class MyGLRenderer implements GLSurfaceView.Renderer
{
    /**
     * Some useful colors...
     */
    static public final float[] black={0.0F,0.0F,0.0F,1.F};
    static public final float[] white={1.0F,1.0F,1.0F,1.F};
    static public final float[] gray={0.5F,0.5F,0.5F,1.F};
    static public final float[] lightgray={0.8F,0.8F,0.8F,1.F};
    static public final float[] darkgray={0.2F,0.2F,0.2F,1.F};
    static public final float[] red={1.F,0.F,0.F,1.F};
    static public final float[] green={0.F,1.F,0.F,1.F};
    static public final float[] blue={0.F,0.F,1.F,1.F};
    static public final float[] yellow={1.F,1.F,0.F,1.F};
    static public final float[] magenta={1.F,0.F,1.F,1.F};
    static public final float[] cyan={0.F,1.0F,1.F,1.F};
    static public final float[] orange={1.F,0.5F,0.F,1.F};
    /**
     * Reference to the scene environment
     */
    private Scene scene;
    /**
     * Reference to the OpenGL surface view
     */
    private GLSurfaceView view;
    /**
     * Shaders
     */

    private MultipleLightingShaders shaders;

    private DepthShader depthShader;


    /**
     * Projection matrix to provide to the shader
     */
    private final float[] projectionmatrix = new float[16];
    private float[] lightSpaceMatrix = new float[16];

    /**
     * @return the current Shader
     */
    public MultipleLightingShaders getShaders()
    {
        return this.shaders;
    }

    public DepthShader getShadowShader() {
        return depthShader;
    }

    /**
     * @return the scene environment
     */
    public Scene getScene() { return this.scene; }

    /**
     * @return the surface view
     */
    public GLSurfaceView getView()
    {
        return this.view;
    }


    /**
     * Constructor
     * @param view OpenGL surface view
     * @param scene the scene environment
     */
    public MyGLRenderer(final GLSurfaceView view, final Scene scene)
    {
        this.view=view;
        this.scene =scene;
        this.shaders=null;
    }

    /**
     * general routine called when the support drawing function is created
     * Aims at initializing all graphics data
     * @param unused
     * @param config
     */
    @Override
    public void onSurfaceCreated(GL10 unused, EGLConfig config)
    {
        // Create shader
        this.depthShader = new DepthShader(this.view.getContext());
        this.shaders=new ShadowShaders(this.view.getContext()); // or other shaders
        ShaderManager.getInstance().getShaders().clear();
        ShaderManager.getInstance().addShaders(this.shaders);
        ShaderManager.getInstance().setDepthShader(depthShader);
        checkGlError("Shader Creation");

        scene.initGraphics(this);
    }


    /**
     * Called on newdisplay events
     * @param unused
     */
    @Override
    public void onDrawFrame(GL10 unused)
    {

        // Display the scene:
        // Drawing the scene is mandatory, since display buffers are swapped in any case.
        renderShadowMap(scene.light2.getCompotent(Light.class));
        renderScene(scene);
        // Dirty mode, so post a new display request to loop
        this.view.requestRender();
    }

    /**
     * Called when the surface has changed (screen rotation, for instance)
     * always called at the beginning, before first display.
     * @param unused
     * @param width
     * @param height
     */
    @Override
    public void onSurfaceChanged(GL10 unused,final int width,final int height) {
        // Adjust the viewport based on geometry changes,
        GLES20.glViewport(0, 0, width, height);
        generateShadowFBO();
        // Compute projection matrix
        float ratio = (float) width / height;
        if (width > height) // Landscape mode
        {
            Matrix.perspectiveM(this.projectionmatrix, 0, 60.F, ratio, 0.1F, 100.F);
        } else // Portrait mode
        {
            Matrix.perspectiveM(this.projectionmatrix, 0, 45.F, ratio, 0.1F, 100.F);
        }
        shaders.setProjectionMatrix(this.projectionmatrix);

    }

    /**
     * Utility method for debugging OpenGL calls.
     * If the operation is not successful, the check throws an error.
     *
     * @param gloperation - Name of the OpenGL call that was called
     */
    public static void checkGlError(String gloperation)
    {
        int firsterror,error;

        // Check if there is an error
        error = GLES20.glGetError();
        if (error==GLES20.GL_NO_ERROR) return;

        // In case of error, display the error list and throw an exception...
        firsterror=error;
        do
        {
            MainActivity.log("Gl Error "+error+" after "+gloperation);
            error = GLES20.glGetError();
        } while (error!=GLES20.GL_NO_ERROR);
        throw new RuntimeException("GL Error "+firsterror+" after "+gloperation);

    }

    /**
     * Utility method to load a texture defined as a resource
     * This method is freely inspired from www.learnopengles.com
     * @param context Context of application
     * @param resourceId Id of the application resource to load
     *                   Typically, this resource is something like R.drawable.name_of_the_file
     * @return Texture handle
     */
    public static int loadTexture(final Context context,final int resourceId)
    {
        // Create a new texture handle to store the loaded texture
        final int[] textureHandle = new int[1];
        GLES20.glGenTextures(1, textureHandle, 0);

        if (textureHandle[0] != 0)
        {
            MainActivity.log("test texture");
            // Load the given ressource as a bitmap
            final BitmapFactory.Options options = new BitmapFactory.Options();
            options.inScaled = false;   // No pre-scaling

            // Read the resource
            final Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), resourceId, options);

            // Bind to the allocated texture handle so that the following instructions are done
            // within this texture handle
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureHandle[0]);

            // Set filtering parameters (can be changed to allow a better visualization)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

            // Load the bitmap into the bound texture
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);

            // Unbind texture
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D,0);

            // Recycle the bitmap, it has been loaded into the graphics card memory and is no longer
            // used in the main memory
            bitmap.recycle();
        }
        return textureHandle[0];
    }


    int[] fboId;
    int[] depthTextureId;
    int[] renderTextureId;

    public void generateShadowFBO() {
        final int SHADOW_WIDTH = 2048, SHADOW_HEIGHT = 2048;

        fboId = new int[1];
        depthTextureId = new int[1];
        renderTextureId = new int[1];

        // create a framebuffer object
        GLES20.glGenFramebuffers(1, fboId, 0);

        // create render buffer and bind 16-bit depth buffer
        GLES20.glGenRenderbuffers(1, depthTextureId, 0);
        GLES20.glBindRenderbuffer(GLES20.GL_RENDERBUFFER, depthTextureId[0]);
        GLES20.glRenderbufferStorage(GLES20.GL_RENDERBUFFER, GLES20.GL_DEPTH_COMPONENT16, SHADOW_WIDTH, SHADOW_HEIGHT);

        // Try to use a texture depth component
        GLES20.glGenTextures(1, renderTextureId, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, renderTextureId[0]);

        // GL_LINEAR does not make sense for depth texture. However, next tutorial shows usage of GL_LINEAR and PCF. Using GL_NEAREST
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);

        // Remove artifact on the edges of the shadowmap
        GLES20.glTexParameteri( GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri( GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId[0]);

        // Use a depth texture
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_DEPTH_COMPONENT, SHADOW_WIDTH, SHADOW_HEIGHT, 0, GLES20.GL_DEPTH_COMPONENT, GLES20.GL_UNSIGNED_INT, null);

        // Attach the depth texture to FBO depth attachment point
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_DEPTH_ATTACHMENT, GLES20.GL_TEXTURE_2D, renderTextureId[0], 0);


        // check FBO status
        int FBOstatus = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER);
        if(FBOstatus != GLES20.GL_FRAMEBUFFER_COMPLETE) {
            MainActivity.log("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use FBO");
            throw new RuntimeException("GL_FRAMEBUFFER_COMPLETE failed, CANNOT use FBO");
        }
    }

    private void renderShadowMap(Light light) {
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboId[0]);

        GLES20.glViewport(0,0,2048,2048);

        //GLES20.glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        GLES20.glClear( GLES20.GL_DEPTH_BUFFER_BIT | GLES20.GL_COLOR_BUFFER_BIT);

        float[] lightProjection = new float[16];
        Matrix.orthoM(lightProjection,0,-10.f,10.f,-10.f,10.f,0.1f,50.f);
        float[] lightView = new float[16];
        float[] lightpos = light.getPos(light.getTransform().getParentModelViewMatrix());
        float[] lightdir = light.getDir(light.getTransform().getParentModelViewMatrix());
        Matrix.setLookAtM(lightView,0,lightpos[0], lightpos[1], lightpos[2], lightpos[0]+lightdir[0], lightpos[1] + lightdir[1],lightpos[2]+lightdir[2],0.f,1.f,0.f);
        lightSpaceMatrix = new float[16];
        Matrix.multiplyMM(lightSpaceMatrix,0,lightProjection,0,lightView,0);

        depthShader.use();
        depthShader.setProjectionMatrix(lightProjection);
        depthShader.setModelViewMatrix(lightView);
        depthShader.setViewMatrix(lightView);

        //j'utilise à la fois la technique de générer l'ombre avec les faces internes pour supprimer
        // l'acné d'ombre sur les objets pleins (cube, sphere, tore, capsule ...) et le biais pour le retirer sur les surfaces planes
        GLES20.glCullFace(GLES20.GL_FRONT);
        scene.update();
        GLES20.glCullFace(GLES20.GL_BACK);
    }

    private void renderScene(Scene scene){
        this.scene.step();
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        GLES20.glViewport(0,0,view.getWidth(),view.getHeight());
        this.shaders.use();
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, renderTextureId[0]);
        shaders.setLightSpaceMatrix(lightSpaceMatrix);
        shaders.setDepthMap(1);
        GLES20.glFrontFace(GLES20.GL_CW);
        scene.setUpReflexionMatrix(this);
        scene.earlyUpdate();
        scene.lateUpdate();
        GLES20.glFrontFace(GLES20.GL_CCW);
        scene.setUpMatrix(this);
        scene.earlyUpdate();
        scene.lateUpdate();
    }
}
