package fr.algo3d.model;

import fr.algo3d.model.models.Color;

/**
 * Class to stock informations about colors and how a model should be rendered (transparency and reflection).
 */
public class Material {
    /**
     * Color.
     */
    private Color color;

    /**
     * Specular color.
     */
    private Color specColor;

    /**
     * Shininess.
     */
    private float shininess;

    /**
     * Transparency.
     */
    private float transparency;

    /**
     * Reflection.
     */
    private float reflection;

    /**
     * Refract index.
     */
    private float refractIndex;

    /**
     * Constructor.
     * By default, transparency and reflection are set to 0 and refract index to 1.
     * @param color
     * @param specColor
     * @param shininess
     */
    public Material(Color color, Color specColor, float shininess) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
        transparency = 0;
        reflection = 0;
        refractIndex = 1;
    }

    /**
     * Constructor.
     * @param color
     * @param specColor
     * @param shininess
     * @param transparency
     * @param reflection
     * @param refractIndex
     */
    public Material(Color color, Color specColor, float shininess, float transparency, float reflection, float refractIndex) {
        this.color = color;
        this.specColor = specColor;
        this.shininess = shininess;
        this.transparency = transparency;
        this.reflection = reflection;
        this.refractIndex = refractIndex;
    }

    /**
     * Returns the color.
     * @return
     */
    public Color getColor() {
        return color;
    }

    /**
     * Returns the specular color.
     * @return
     */
    public Color getSpecColor() {
        return specColor;
    }

    /**
     * Returns the shininess.
     * @return
     */
    public float getShininess() {
        return shininess;
    }

    /**
     * Returns the transparency.
     * @return
     */
    public float getTransparency() {
        return transparency;
    }

    /**
     * Returns the reflection.
     * @return
     */
    public float getReflection() {
        return reflection;
    }

    /**
     * Returns the refract index.
     * @return
     */
    public float getRefractIndex() {
        return refractIndex;
    }
}
