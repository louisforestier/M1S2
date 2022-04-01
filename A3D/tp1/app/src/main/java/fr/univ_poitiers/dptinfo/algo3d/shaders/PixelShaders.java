package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

public class PixelShaders extends LightingShaders{

    /**
     * Constructor. nothing to do, everything is done in the super class...
     *
     * @param context
     */
    public PixelShaders(Context context) {
        super(context);
    }


    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"pixel_vert.glsl","pixel_frag.glsl");
    }
}
