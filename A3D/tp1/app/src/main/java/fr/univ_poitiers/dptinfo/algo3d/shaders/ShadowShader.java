package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

public class ShadowShader extends BasicShaders{


    /**
     * Constructor of the complete rendering Shader programs
     *
     * @param context
     */
    public ShadowShader(Context context) {
        super(context);
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context,"shadow_vert.glsl","shadow_frag.glsl");
    }
}
