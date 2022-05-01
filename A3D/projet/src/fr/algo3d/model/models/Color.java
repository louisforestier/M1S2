package fr.algo3d.model.models;

/**
 * Class to represent colors with floats between 0 and 1.
 */
public class Color {
    /**
     * Value of the red channel.
     */
    private float r;

    /**
     * Value of the green channel.
     */
    private float g;

    /**
     * Value of the blue channel.
     */
    private float b;

    static public final Color black=new Color(0.0F,0.0F,0.0F);
    static public final Color white= new Color(1.0F,1.0F,1.0F);
    static public final Color gray=new Color(0.5F,0.5F,0.5F);
    static public final Color lightgray= new Color(0.8F,0.8F,0.8F);
    static public final Color darkgray=new Color(0.2F,0.2F,0.2F);
    static public final Color red=new Color(1.F,0.F,0.F);
    static public final Color green=new Color(0.F,1.F,0.F);
    static public final Color blue=new Color(0.F,0.F,1.F);
    static public final Color yellow=new Color(1.F,1.F,0.F);
    static public final Color magenta=new Color(1.F,0.F,1.F);
    static public final Color cyan=new Color(0.F,1.0F,1.F);
    static public final Color orange=new Color(1.F,0.5F,0.F);

    /**
     * Default constructor.
     * Create a black color.
     */
    public Color() {
    }

    /**
     * Constructor.
     * @param r
     * @param g
     * @param b
     */
    public Color(float r, float g, float b) {
        this.r = r;
        this.g = g;
        this.b = b;
    }

    /**
     * Add a color and the current object and returns the result.
     * @param c
     * @return
     */
    public Color add(Color c){
        float newR = Math.min(this.r + c.r,1.f);
        float newG = Math.min(this.g + c.g,1.f);
        float newB = Math.min(this.b + c.b,1.f);
        return new Color(newR, newG, newB);
    }

    /**
     * Multiply the current object and a color and returns the result.
     * @param c
     * @return
     */
    public Color mul(Color c){
        float newR = this.r * c.r;
        float newG = this.g * c.g;
        float newB = this.b * c.b;
        return new Color(newR,newG,newB);
    }

    /**
     * Multiply the current object and a float and returns the result.
     * @param s
     * @return
     */
    public Color scale(float s){
        float newR = this.r * s;
        float newG = this.g * s;
        float newB = this.b * s;
        return new Color(newR,newG,newB);
    }

    /**
     * Returns the red channel.
     * @return
     */
    public float getR() {
        return r;
    }

    /**
     * Returns the green channel.
     * @return
     */
    public float getG() {
        return g;
    }

    /**
     * Returns the blue channel.
     * @return
     */
    public float getB() {
        return b;
    }
}
