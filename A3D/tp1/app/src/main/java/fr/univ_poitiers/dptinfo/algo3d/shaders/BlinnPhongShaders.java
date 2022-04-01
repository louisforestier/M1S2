package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

public class BlinnPhongShaders extends LightingShaders {

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */
    public BlinnPhongShaders(Context context) {
        super(context);
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context, "blinn_phong_vert.glsl", "blinn_phong_frag.glsl");
    }

}
