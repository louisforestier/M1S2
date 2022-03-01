package fr.univ_poitiers.dptinfo.algo3d.objimporter;

public class OBJVertex {
    private int v;
    private int vt;
    private int vn;

    public OBJVertex(String data) {
        String[] index = data.split("/");
        this.v = Integer.parseInt(index[0]);
        if (index.length > 1) {
            if (!index[1].isEmpty())
                this.vt = Integer.parseInt(index[1]);
            this.vn = Integer.parseInt(index[2]);
        }
    }

    public int getV() {
        return v;
    }

    public int getVt() {
        return vt;
    }

    public int getVn() {
        return vn;
    }
}
