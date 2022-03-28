package fr.algo3d;

import fr.algo3d.models.Color;

public class Material {
    private Color color;
    private Color specColor;
    private float shininess;

    public Material(Color color, Color specColor, float shininess) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
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
}
