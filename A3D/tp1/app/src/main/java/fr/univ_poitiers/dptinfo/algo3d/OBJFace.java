package fr.univ_poitiers.dptinfo.algo3d;

public class OBJFace {

    private OBJVertex v1;
    private OBJVertex v2;
    private OBJVertex v3;

    public OBJFace(String[] data) {
        this.v1 = new OBJVertex(data[1]);
        this.v2 = new OBJVertex(data[2]);
        this.v3 = new OBJVertex(data[3]);
    }

    public OBJVertex getV1() {
        return v1;
    }

    public OBJVertex getV2() {
        return v2;
    }

    public OBJVertex getV3() {
        return v3;
    }
}
