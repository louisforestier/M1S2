package fr.univ_poitiers.dptinfo.algo3d.shaders;


import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class ShaderManager {

    private Map<Class<? extends MultipleLightingShaders>,MultipleLightingShaders> shaders = new HashMap<>();
    private static ShaderManager INSTANCE;
    private ShaderManager() {
    }

    public static ShaderManager getInstance(){
        if (INSTANCE == null)
            INSTANCE = new ShaderManager();
        return INSTANCE;
    }

    public  MultipleLightingShaders getShader(Class<? extends MultipleLightingShaders> type){
        return shaders.get(type);
    }


    public Map<Class<? extends MultipleLightingShaders>, MultipleLightingShaders> getShaders() {
        return shaders;
    }

    public void addShaders(MultipleLightingShaders shaders){
        this.shaders.put(shaders.getClass(),shaders);
    }
}
