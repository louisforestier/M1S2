package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;

public class DepthShader extends BasicShaders {


    /**
     * Constructor of the complete rendering Shader programs
     *
     * @param context
     */
    public DepthShader(Context context) {
        super(context);
    }

    @Override
    public int createProgram(Context context) {
        return initializeShadersFromResources(context, "depth_vert.glsl", "depth_frag.glsl");
    }
}
