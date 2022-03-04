package fr.univ_poitiers.dptinfo.algo3d;

public class Light {
    private LightType type;
    private float[] ambient;
    private float[] diffuse;
    private float[] specular;
    private float constant;
    private float linear;
    private float quadratic;

    public Light(LightType type) {
        this.type = type;
        ambient =new float[]{0.2f,0.2f,0.2f,1.f};
        diffuse = new float[]{0.8f,0.8f,0.8f,1.f};
        specular = new float[]{0.8f,0.8f,0.8f,1.f};
        constant = 1.f ;
        linear = 0.09f;
        quadratic = 0.032f;
    }

    public Light(float[] ambient, float[] diffuse, float[] specular, float constant, float linear, float quadratic) {
        this.ambient = ambient;
        this.diffuse = diffuse;
        this.specular = specular;
        this.constant = constant;
        this.linear = linear;
        this.quadratic = quadratic;
    }

    public LightType getType() {
        return type;
    }

    public void setType(LightType type) {
        this.type = type;
    }

    public float[] getAmbient() {
        return ambient;
    }

    public void setAmbient(float[] ambient) {
        this.ambient = ambient;
    }

    public float[] getDiffuse() {
        return diffuse;
    }

    public void setDiffuse(float[] diffuse) {
        this.diffuse = diffuse;
    }

    public float[] getSpecular() {
        return specular;
    }

    public void setSpecular(float[] specular) {
        this.specular = specular;
    }

    public float getConstant() {
        return constant;
    }

    public void setConstant(float constant) {
        this.constant = constant;
    }

    public float getLinear() {
        return linear;
    }

    public void setLinear(float linear) {
        this.linear = linear;
    }

    public float getQuadratic() {
        return quadratic;
    }

    public void setQuadratic(float quadratic) {
        this.quadratic = quadratic;
    }
}
