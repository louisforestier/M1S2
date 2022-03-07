package fr.univ_poitiers.dptinfo.algo3d.mesh;

import android.opengl.GLES20;
import android.opengl.Matrix;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import fr.univ_poitiers.dptinfo.algo3d.MyGLRenderer;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.Component;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.GameObject;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.Transform;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;

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
    public void lateUpdate() {
        if (gameObject.getCompotent(MeshFilter.class) != null) {
            float[] modelviewmatrix = new float[16];
            Matrix.multiplyMM(modelviewmatrix, 0, material.getShader().getViewMatrix(), 0, transform.getModelViewMatrix(), 0);
            material.getShader().setModelViewMatrix(modelviewmatrix);
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
