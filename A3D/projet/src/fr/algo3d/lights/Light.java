package fr.algo3d.lights;


import fr.algo3d.models.Color;
import fr.algo3d.models.Vec3f;


public class Light {
    private Vec3f position;
    private Color ambient;
    private Color diffuse;
    private Color specular;

    public Light(Vec3f position, float[] ambient, float[] diffuse, float[] specular) {
        this.position = position;
        this.ambient = new Color(ambient);
        this.diffuse = new Color(diffuse);
        this.specular = new Color(specular);
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
