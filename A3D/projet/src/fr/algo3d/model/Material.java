package fr.algo3d.model;

import fr.algo3d.model.models.Color;

public class Material {
    private Color color;
    private Color specColor;
    private float shininess;
    private float transparency;
    private float reflection;
    private float refractIndex;

    public Material(Color color, Color specColor, float shininess) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
        transparency = 0;
        reflection = 0;
        refractIndex = 0;
    }

    public Material(Color color, Color specColor, float shininess, float transparency, float reflection, float refractIndex) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
        this.transparency = transparency;
        this.reflection = reflection;
        this.refractIndex = refractIndex;
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

    public float getRefractIndex() {
        return refractIndex;
    }
}
