package fr.univ_poitiers.dptinfo.algo3d.shaders;

public enum LightType {
    POINT(0),
    DIRECTIONAL(1),
    SPOT(2);

    private final int value;

    LightType(int value) {
        this.value = value;
    }

    public int getValue() {
        return value;
    }
}
