package fr.univ_poitiers.dptinfo.algo3d.mesh;

import fr.univ_poitiers.dptinfo.algo3d.gameobject.Component;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.GameObject;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.Transform;

public class MeshFilter extends Component {

    private Mesh mesh;

    public MeshFilter(GameObject gameObject, Transform transform) {
        super(gameObject, transform);
    }

    public Mesh getMesh() {
        return mesh;
    }

    public void setMesh(Mesh mesh) {
        this.mesh = mesh;
    }

}
