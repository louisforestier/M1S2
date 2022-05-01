package fr.algo3d.model.models;

import fr.algo3d.model.Material;

/**
 * Abstract class to represent the model to be rendered.
 */
public abstract class Model {

    /**
     * Material of the model.
     */
    private Material material;

    /**
     * Constructor.
     * @param material
     */
    public Model(Material material) {
        this.material = material;
    }

    /**
     * Returns the closest intersection lambda.
     * @param P
     * @param v
     * @return the closest interesection lambda or -1 if there is none.
     */
    public abstract float getIntersection (Vec3f P, Vec3f v);

    /**
     * Returns the normal at the point i.
     * @param i
     * @return
     */
    public abstract Vec3f getNormal(Vec3f i);

    /**
     * Returns the material diffuse color.
     * @return
     */
    public Color getColor() {
        return material.getColor();
    }

    /**
     * Returns the material specular color.
     * @return
     */
    public Color getSpecularMaterial() {
        return material.getSpecColor();
    }

    /**
     * Returns the material shininess.
     * @return
     */
    public float getShininess(){
        return material.getShininess();
    }

    /**
     * Returns the material transparency kt.
     * @return
     */
    public float getTransparency(){
        return material.getTransparency();
    }

    /**
     * Returns the material reflection kr.
     * @return
     */
    public float getReflection(){
        return material.getReflection();
    }

    /**
     * Returns the material refract index.
     * @return
     */
    public float getRefractIndex() {
        return material.getRefractIndex();
    }

}
