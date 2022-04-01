package fr.univ_poitiers.dptinfo.algo3d.shaders;


import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class ShaderManager {

    private static boolean render;

    private Map<Class<? extends MultipleLightingShaders>,MultipleLightingShaders> shaders = new HashMap<>();
    private DepthShader depthShader;
    private static ShaderManager INSTANCE;
    private ShaderManager() {
    }

    public static ShaderManager getInstance(){
        if (INSTANCE != null)
            return INSTANCE;
        synchronized (ShaderManager.class){
            if (INSTANCE == null)
                INSTANCE = new ShaderManager();
        }
        return INSTANCE;
    }

    public  MultipleLightingShaders getShader(Class<? extends MultipleLightingShaders> type){
        return shaders.get(type);
    }

    public static boolean isRender() {
        return render;
    }

    public static void setRender(boolean render) {
        ShaderManager.render = render;
    }

    public Map<Class<? extends MultipleLightingShaders>, MultipleLightingShaders> getShaders() {
        return shaders;
    }

    public void addShaders(MultipleLightingShaders shaders){
        this.shaders.put(shaders.getClass(),shaders);
    }

    public DepthShader getDepthShader() {
        return depthShader;
    }

    public void setDepthShader(DepthShader depthShader) {
        this.depthShader = depthShader;
    }
}
