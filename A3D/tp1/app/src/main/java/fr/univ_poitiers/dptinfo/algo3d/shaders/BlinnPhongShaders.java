package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

/**
 * Shader class to use lightning with the blinn phong formula.
 */
public class BlinnPhongShaders extends LightingShaders {

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */
    public BlinnPhongShaders(Context context) {
        super(context);
    }

    /**
     * Create the shader program with the blinn_phong glsl.
     * @param context - context of the application
     * @return the shader program handle
     */
    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context, "blinn_phong_vert.glsl", "blinn_phong_frag.glsl");
    }

}
