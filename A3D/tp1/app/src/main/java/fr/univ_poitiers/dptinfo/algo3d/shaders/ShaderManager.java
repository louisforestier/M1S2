package fr.univ_poitiers.dptinfo.algo3d.shaders;


import java.util.HashSet;
import java.util.Set;

public class ShaderManager {

    private Set<MultipleLightingShaders> shaders = new HashSet<>();
    private static ShaderManager INSTANCE;
    private ShaderManager() {
    }

    public static ShaderManager getInstance(){
        if (INSTANCE == null)
            INSTANCE = new ShaderManager();
        return INSTANCE;
    }

    public  <T extends BasicShaders> T getShader(Class<T> type){
        for (BasicShaders shader : shaders){
            if (type.isInstance(shader))
                return type.cast(shader);

        }
        return null;
    }

    public Set<MultipleLightingShaders> getShaders(){
        return shaders;
    }

    public void addShaders(MultipleLightingShaders shaders){
        this.shaders.add(shaders);
    }
}
