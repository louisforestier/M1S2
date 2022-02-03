package fr.univ_poitiers.dptinfo.algo3d;

public class Tictac extends Mesh {

    public Tictac(int slice, int quarter) {
        int verticesSize;
        int trianglesSize;
        if (slice % 2 == 0){
           verticesSize = ((slice) * (quarter + 1) + 2) * 3;
           trianglesSize = (quarter * (slice - 1) * 2 + quarter * 2) * 3;
        } else {
            verticesSize = ((slice-1) * (quarter + 1) + 2) * 3;
            trianglesSize = (quarter * (slice - 2) * 2 + quarter * 2) * 3;
        }
        vertexpos = new float[verticesSize];
        triangles = new int[trianglesSize];
        float r = 1.f;
        int k = 0;
        for (int i = 1; i <= slice/2; i++) {
            double theta = Math.toRadians(90.0 - (180.0 / slice) * i);
            for (int j = 0; j <= quarter; j++) {
                double phi = Math.toRadians((360.0 / quarter) * j);
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.cos(phi));
                vertexpos[k++] = (float) (r * Math.sin(theta));
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.sin(phi));
            }
        }
        int step;
        if (slice % 2 == 0){
            step = slice / 2;
        } else {
            step = slice/2 +1;
        }
        double tmptheta = Math.toRadians(90.0 - (180.0 / slice) * step);
        for (int j = 0; j <= quarter; j++) {
            double phi = Math.toRadians((360.0 / quarter) * j);
            vertexpos[k++] = (float) (r * Math.cos(tmptheta) * Math.cos(phi));
            vertexpos[k++] = (float) (r * Math.sin(tmptheta)) - 1.f;
            vertexpos[k++] = (float) (r * Math.cos(tmptheta) * Math.sin(phi));
        }

        for (int i = step+1; i < slice; i++) {
            double theta = Math.toRadians(90.0 - (180.0 / slice) * i);
            for (int j = 0; j <= quarter; j++) {
                double phi = Math.toRadians((360.0 / quarter) * j);
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.cos(phi));
                vertexpos[k++] = (float) (r * Math.sin(theta)) - 1.f;
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.sin(phi));
            }
        }
        vertexpos[vertexpos.length - 5] = -1;
        vertexpos[vertexpos.length - 2] = 1;

        k = 0;
        for (int i = 0; i < slice - 1; i++) {
            for (int j = 0; j < quarter; j++) {
                triangles[k++] = i * (quarter + 1) + j;
                triangles[k++] = i * (quarter + 1) + 1 + j;
                triangles[k++] = i * (quarter + 1) + quarter + 2 + j;
                triangles[k++] = i * (quarter + 1) + j;
                triangles[k++] = i * (quarter + 1) + quarter + 2 + j;
                triangles[k++] = i * (quarter + 1) + quarter + 1 + j;
            }
        }
        for (int i = 0; i < quarter; i++) {
            triangles[k++] = vertexpos.length / 3 - 1;
            triangles[k++] = i + 1;
            triangles[k++] = i;
        }
        for (int i = 0; i < quarter; i++) {
            triangles[k++] = vertexpos.length / 3 - 2;
            triangles[k++] = i - 1 + vertexpos.length / 3 - 2 - quarter;
            triangles[k++] = i + vertexpos.length / 3 - 2 - quarter;
        }

    }
}
