package fr.univ_poitiers.dptinfo.algo3d.gameobject;

public abstract class Component {

    protected GameObject gameObject;
    protected Transform transform;

    public Component(GameObject gameObject, Transform transform) {
        this.gameObject = gameObject;
        this.transform = transform;
    }

    public fr.univ_poitiers.dptinfo.algo3d.gameobject.Transform getTransform() {
        return transform;
    }

    public void start(){}
    public void earlyUpdate(){}
    public void update(){}
    public void lateUpdate(){}
}
