package fr.algo3d.model.lights;


import fr.algo3d.model.models.Color;
import fr.algo3d.model.models.Vec3f;

/**
 * Class to represent lights.
 */
public class Light {

    /**
     * Position.
     */
    private Vec3f position;

    /**
     * Diffuse color.
     */
    private Color diffuse;

    /**
     * Specular color.
     */
    private Color specular;

    /**
     * Contructor.
     * @param position
     * @param diffuse
     * @param specular
     */
    public Light(Vec3f position, Color diffuse, Color specular) {
        this.position = position;
        this.diffuse = diffuse;
        this.specular = specular;
    }

    /**
     * Returns the position.
     * @return position
     */
    public Vec3f getPosition() {
        return position;
    }

    /**
     * Returns the diffuse color.
     * @return diffuse color
     */
    public Color getDiffuse() {
        return diffuse;
    }

    /**
     * Returns the specular color.
     * @return specular color
     */
    public Color getSpecular() {
        return specular;
    }
}
