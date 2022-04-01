package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

public class BlinnPhongMultipleLightShaders extends MultipleLightingShaders {

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */

    public BlinnPhongMultipleLightShaders(Context context) {
        super(context);
    }

    @Override
    public void findVariables() {
        super.findVariables();
    }

    @Override
    public boolean useTypeLight() {
        return true;
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context, "blinn_phong_with_multiple_lights_vert.glsl", "blinn_phong_with_multiple_lights_frag.glsl");
    }

}
