package fr.algo3d.models;

import fr.algo3d.Material;

public abstract class Model {

    private Material material;


    public Model(Material material) {
        this.material = material;
    }

    public abstract float getIntersection (Vec3f P, Vec3f v);

    public abstract Vec3f getNormal(Vec3f i);

    public Color getDiffuseMaterial() {
        return material.getColor();
    }
    public Color getAmbiantMaterial() {
        return material.getColor();
    }
    public Color getSpecularMaterial() {
        return material.getSpecColor();
    }

    public float getShininess(){
        return material.getShininess();
    }
    public float getTransparency(){
        return material.getTransparency();
    }
    public float getReflection(){
        return material.getReflection();
    }
    public float getRefractIndex() {
        return material.getRefractIndex();
    }

}
