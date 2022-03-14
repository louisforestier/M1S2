package fr.univ_poitiers.dptinfo.algo3d.mesh;

import android.opengl.Matrix;

import fr.univ_poitiers.dptinfo.algo3d.gameobject.Component;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.GameObject;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.Transform;
import fr.univ_poitiers.dptinfo.algo3d.shaders.ShaderManager;

public class MeshRenderer extends Component {

    private Material material;


    public MeshRenderer(GameObject gameObject, Transform transform) {
        super(gameObject, transform);
    }

    public Material getMaterial() {
        return material;
    }

    public void setMaterial(Material material) {
        this.material = material;
    }

    @Override
    public void start() {
        gameObject.getCompotent(MeshFilter.class).getMesh().initGraphics();
    }


    @Override
    public void update() {
        renderShadow();
    }

    private void renderShadow() {
        if (gameObject.getCompotent(MeshFilter.class) != null) {
            float[] modelviewmatrix = new float[16];
            Matrix.multiplyMM(modelviewmatrix, 0, ShaderManager.getInstance().getDepthShader().getViewMatrix(), 0, transform.getModelViewMatrix(), 0);
            ShaderManager.getInstance().getDepthShader().setModelViewMatrix(modelviewmatrix);
            gameObject.getCompotent(MeshFilter.class).getMesh().draw(ShaderManager.getInstance().getDepthShader());
        }
    }


    @Override
    public void lateUpdate() {
        render();
    }

    private void render() {
        if (gameObject.getCompotent(MeshFilter.class) != null) {
            float[] modelviewmatrix = new float[16];
            Matrix.multiplyMM(modelviewmatrix, 0, material.getShader().getViewMatrix(), 0, transform.getModelViewMatrix(), 0);
            material.getShader().setModelViewMatrix(modelviewmatrix);
            material.getShader().setModelMatrix(transform.getModelViewMatrix());
            material.update();
            switch (material.getDrawMode()) {
                case TRIANGLES:
                    gameObject.getCompotent(MeshFilter.class).getMesh().draw(material.getShader());
                    break;
                case WIREFRAME:
                    gameObject.getCompotent(MeshFilter.class).getMesh().drawLinesOnly(material.getShader());
                    break;
                case TRIANGLES_AND_WIREFRAME:
                    gameObject.getCompotent(MeshFilter.class).getMesh().drawWithLines(material.getShader());
            }
        }
    }


}
