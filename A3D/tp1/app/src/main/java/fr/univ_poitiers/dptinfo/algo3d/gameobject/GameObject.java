package fr.univ_poitiers.dptinfo.algo3d.gameobject;

import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.List;

import fr.univ_poitiers.dptinfo.algo3d.mesh.Material;
import fr.univ_poitiers.dptinfo.algo3d.mesh.Mesh;
import fr.univ_poitiers.dptinfo.algo3d.mesh.MeshFilter;
import fr.univ_poitiers.dptinfo.algo3d.mesh.MeshRenderer;
import fr.univ_poitiers.dptinfo.algo3d.shaders.MultipleLightingShaders;

public class GameObject {

    private Transform transform;
    private List<GameObject> children = new ArrayList<>();
    private List<Component> components = new ArrayList<>();
    private GameObject parent = null;


    public GameObject(){
        this.transform = new Transform(this);
    }

    public void setMesh(Mesh mesh) {
        addComponent(MeshFilter.class);
        getCompotent(MeshFilter.class).setMesh(mesh);
    }

    public void addMeshRenderer( Material material) {
        addComponent(MeshRenderer.class);
        getCompotent(MeshRenderer.class).setMaterial(material);
    }

    public <T extends Component> void addComponent(Class<T> type){
        try {
            components.add(type.getDeclaredConstructor(GameObject.class,Transform.class).newInstance(this,transform));
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        }
    }

    public <T extends Component> T getCompotent(Class<T> type){
        for (Component c : components){
            if (type.isInstance(c))
                return type.cast(c);
        }
        return null;
    }


    public Transform getTransform() {
        return transform;
    }

    public void addChildren(GameObject child){
        this.children.add(child);
        child.parent = this;
    }

    public GameObject getParent() {
        return parent;
    }

    public void start(){
        for (Component c : components)
            c.start();
        if (this.children.size() > 0){
            for (GameObject go : this.children) {
                go.start();
            }
        }
    }

    public void update(){
        for (Component c : components) {
            c.update();
        }
        if (this.children.size() > 0){
            for (GameObject go : this.children) {
                go.update();
            }
        }
    }

    public void lateUpdate(){
        for (Component c : components){
            c.lateUpdate();
        }
        if (this.children.size() > 0){
            for (GameObject go : this.children) {
                go.lateUpdate();
            }
        }
    }
}
