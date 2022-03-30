package fr.algo3d.model.lights;


import fr.algo3d.model.models.Color;
import fr.algo3d.model.models.Vec3f;


public class Light {
    private Vec3f position;
    private Color ambient;
    private Color diffuse;
    private Color specular;

    public Light(Vec3f position, Color ambient, Color diffuse, Color specular) {
        this.position = position;
        this.ambient = ambient;
        this.diffuse = diffuse;
        this.specular = specular;
    }


    public Vec3f getPosition() {
        return position;
    }

    public Color getAmbient() {
        return ambient;
    }

    public Color getDiffuse() {
        return diffuse;
    }

    public Color getSpecular() {
        return specular;
    }
}
