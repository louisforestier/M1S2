package fr.algo3d;

import fr.algo3d.models.Color;

public class Material {
    private Color color;
    private Color specColor;
    private float shininess;
    private float transparency;
    private float reflection;

    public Material(Color color, Color specColor, float shininess) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
        transparency = 0;
        reflection = 0;
    }

    /**
     *
     * @param color
     * @param specColor
     * @param shininess
     * @param transparency float between 0 and 1
     * @param reflection float between 0 and 1
     */
    public Material(Color color, Color specColor, float shininess, float transparency, float reflection) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
        this.transparency = transparency;
        this.reflection = reflection;
    }

    public Color getColor() {
        return color;
    }

    public Color getSpecColor() {
        return specColor;
    }

    public float getShininess() {
        return shininess;
    }

    public float getTransparency() {
        return transparency;
    }

    public float getReflection() {
        return reflection;
    }
}
