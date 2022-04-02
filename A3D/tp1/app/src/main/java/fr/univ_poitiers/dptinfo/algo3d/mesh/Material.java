package fr.univ_poitiers.dptinfo.algo3d.mesh;

import android.opengl.GLES20;

import fr.univ_poitiers.dptinfo.algo3d.MyGLRenderer;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShaderManager;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShadowShaders;

/**
 * Class to store the material properties to be used in Blinn-Phong calculation of color in the shader.
 */
public class Material {

    /**
     * The default shader to use.
     */
    private final static Class<? extends MultipleLightingShaders> defaultShader = ShadowShaders.class;

    /**
     * The ambient/diffuse color of the material.
     */
    private float[] color;

    /**
     * The specular color of the material.
     */
    private float[] specColor;
    /**
     * The shininess of the material.
     */
    private float shininess;
    /**
     * The shader of the material. Should allow the use of different shaders in a Scene, not tested yet.
     */
    private Class<? extends MultipleLightingShaders> shader;
    /**
     * How the object should be drawn. See {@link DrawMode}
     */
    private DrawMode drawMode;

    /**
     * Handle of the texture stored on GPU.
     */
    private int textureId = -1;

    /**
     * Default Constructor.
     * Creates a white material.
     */
    public Material() {
        this.shader = defaultShader;
        color = MyGLRenderer.white;
        specColor = MyGLRenderer.white;
        shininess = 32.f;
        drawMode = DrawMode.TRIANGLES;
    }

    /**
     * Constructor.
     * @param color - the ambient/diffuse color of the material.
     */
    public Material(float[] color) {
        this.color = color;
        this.shader = defaultShader;
        specColor = MyGLRenderer.white;
        shininess = 32.f;
        drawMode = DrawMode.TRIANGLES;
    }

    /**
     * Constructor.
     * @param color - the ambient/diffuse color of the material.
     * @param specColor - the specular color of the material.
     * @param shininess - the shininess of the material.
     */
    public Material(float[] color, float[] specColor, float shininess) {
        this.color = color;
        this.shader = defaultShader;
        this.specColor = specColor;
        this.shininess = shininess;
        drawMode = DrawMode.TRIANGLES;
    }

    public float[] getColor() {
        return color;
    }

    public float[] getSpecColor() {
        return specColor;
    }

    public float getShininess() {
        return shininess;
    }

    public void setTextureId(int textureId) {
        this.textureId = textureId;
    }

    public void setShader(Class<? extends MultipleLightingShaders> shader) {
        this.shader = shader;
    }

    public MultipleLightingShaders getShader() {
        return ShaderManager.getInstance().getShader(shader);
    }

    public DrawMode getDrawMode() {
        return drawMode;
    }

    public void update() {
        ShaderManager.getInstance().getShader(shader).setMaterialColor(color);
        ShaderManager.getInstance().getShader(shader).setMaterialSpecular(specColor);
        ShaderManager.getInstance().getShader(shader).setMaterialShininess(shininess);
        ShaderManager.getInstance().getShader(shader).setTexturing(textureId != -1);
        if (textureId != -1) {
            GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
            ShaderManager.getInstance().getShader(shader).setTextureUnit(0);
        }
    }
}
