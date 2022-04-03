package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.opengl.Matrix;

import fr.univ_poitiers.dptinfo.algo3d.gameobject.Component;
import fr.univ_poitiers.dptinfo.algo3d.gameobject.GameObject;

public class Light extends Component {
    private LightType type;
    private float[] position;
    private float[] direction;
    private float[] ambient;
    private float[] diffuse;
    private float[] specular;
    private float constant;
    private float linear;
    private float quadratic;
    private float cutOff;
    private float outerCutOff;

    public Light(GameObject gameObject) {
        super(gameObject);
        this.type = LightType.POINT;
        ambient = new float[]{0.2f, 0.2f, 0.2f, 1.f};
        diffuse = new float[]{0.8f, 0.8f, 0.8f, 1.f};
        specular = new float[]{0.8f, 0.8f, 0.8f, 1.f};
        constant = 1.f;
        linear = 0.09f;
        quadratic = 0.032f;
        cutOff = 12.5f;
        outerCutOff = 17.5f;
    }

    public float[] getPos(final float[] viewmatrix) {
        float[] lightPos = new float[4];
        Matrix.multiplyMV(lightPos, 0, viewmatrix, 0, new float[]{transform.getPosx(), transform.getPosy(), transform.getPosz(), 1.0f}, 0);
        return new float[]{lightPos[0], lightPos[1], lightPos[2]};
    }

    public float[] getDir(final float[] viewmatrix) {
        float[] lightDir = new float[4];
        float[] lightlocalDir = new float[]{
                (float) (Math.cos(Math.toRadians(transform.getRoty())) * Math.cos(Math.toRadians(transform.getRotx()))),
                (float) Math.sin(Math.toRadians(transform.getRotx())),
                (float) (Math.sin(Math.toRadians(transform.getRoty())) * Math.cos(Math.toRadians(transform.getRotx()))),
                0.f
        };
        Matrix.multiplyMV(lightDir, 0, viewmatrix, 0, lightlocalDir, 0);
        return new float[]{lightDir[0], lightDir[1], lightDir[2]};
    }


    public void initLighting(BasicShaders shaders, final float[] modelviewmatrix) {
        position = getPos(modelviewmatrix);
        direction = getDir(modelviewmatrix);
        if (shaders.useTypeLight()) {
            if (shaders instanceof MultipleLightingShaders) {
                MultipleLightingShaders mls = (MultipleLightingShaders) shaders;
                switch (type) {
                    case DIRECTIONAL:
                        mls.setDirLight(this);
                        break;
                    case POINT:
                        mls.setPointLight(this);
                        break;
                    case SPOT:
                        mls.setSpotLight(this);
                        break;
                }
            }
            else {
                BlinnPhongTypeLightShaders tls = (BlinnPhongTypeLightShaders) shaders;
                tls.setLightType(type.getValue());
                tls.setLightPosition(position);
                tls.setLightPosition(position);
                tls.setAmbiantLight(ambient);
                tls.setLightColor(diffuse);
                tls.setLightSpecular(specular);
                tls.setLightAttenuation(constant,linear,quadratic);
                tls.setLightDirection(direction);
                tls.setCutOff(cutOff);
                tls.setOuterCutOff(outerCutOff);
            }
        }
        else {
            if (shaders instanceof LightingShaders){
                LightingShaders ls = (LightingShaders) shaders;
                ls.setLightPosition(position);
                ls.setAmbiantLight(ambient);
                ls.setLightColor(diffuse);
                ls.setLightSpecular(specular);
                ls.setLightAttenuation(constant,linear,quadratic);
            }
        }
    }


    @Override
    public void earlyUpdate() {
        super.earlyUpdate();
        for (MultipleLightingShaders s : ShaderManager.getInstance().getShaders().values()) {
            float[] modelviewmatrix = new float[16];
            Matrix.multiplyMM(modelviewmatrix, 0, s.getViewMatrix(), 0, transform.getParentModelViewMatrix(), 0);
            initLighting(s, modelviewmatrix);
        }
    }

    public LightType getType() {
        return type;
    }

    public void setType(LightType type) {
        this.type = type;
    }

    public float[] getPosition() {
        return position;
    }

    public void setPosition(float[] position) {
        this.position = position;
    }

    public float[] getDirection() {
        return direction;
    }

    public void setDirection(float[] direction) {
        this.direction = direction;
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

    public float getCutOff() {
        return (float) Math.cos(Math.toRadians(cutOff));
    }

    public void setCutOff(float cutOff) {
        this.cutOff = cutOff;
    }

    public float getOuterCutOff() {
        return (float) Math.cos(Math.toRadians(outerCutOff));
    }

    public void setOuterCutOff(float outerCutOff) {
        this.outerCutOff = outerCutOff;
    }
}
