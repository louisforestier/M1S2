package fr.univ_poitiers.dptinfo.algo3d.mesh;

import fr.univ_poitiers.dptinfo.algo3d.MyGLRenderer;
import fr.univ_poitiers.dptinfo.algo3d.shaders.BasicShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.BlinnPhongMultipleLightShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShaderManager;

public class Material {

    private final static Class<? extends MultipleLightingShaders> defaultShader = BlinnPhongMultipleLightShaders.class;

    private float[] color;
    private float[] specColor;
    private float shininess;
    private Class<? extends MultipleLightingShaders> shader;
    private DrawMode drawMode;

    public Material(){
        this.shader = defaultShader;
        color = MyGLRenderer.white;
        specColor = MyGLRenderer.white;
        shininess = 32.f;
        drawMode = DrawMode.TRIANGLES;
    }

    public Material(float[] color) {
        this.color = color;
        this.shader = defaultShader;
        specColor = MyGLRenderer.white;
        shininess = 32.f;
        drawMode = DrawMode.TRIANGLES;
    }

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

    public void setShader(Class<? extends MultipleLightingShaders> shader) {
        this.shader = shader;
    }

    public MultipleLightingShaders getShader(){
        return ShaderManager.getInstance().getShader(shader);
    }

    public DrawMode getDrawMode() {
        return drawMode;
    }

    public void update(){
        ShaderManager.getInstance().getShader(shader).setMaterialColor(color);
        ShaderManager.getInstance().getShader(shader).setMaterialSpecular(specColor);
        ShaderManager.getInstance().getShader(shader).setMaterialShininess(shininess);
    }
}
