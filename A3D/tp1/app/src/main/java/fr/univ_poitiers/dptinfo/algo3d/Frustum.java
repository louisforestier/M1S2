package fr.univ_poitiers.dptinfo.algo3d;

public class Frustum extends Mesh {

    public Frustum(float r1, float r2, int quarter) {

        vertexpos = new float[((2) * (quarter + 1)+2) * 3];
        triangles = new int[quarter * 4 * 3];

        int k = 0;

        for (int i = 0; i <= quarter; i++) {
            double theta = Math.toRadians((360.0 / quarter) * i);
            vertexpos[k++] = (float) (r1 * Math.cos(theta));
            vertexpos[k++] = 0.f;
            vertexpos[k++] = (float) (r1 * Math.sin(theta));
            vertexpos[k++] = (float) (r2 * Math.cos(theta));
            vertexpos[k++] = 1.f;
            vertexpos[k++] = (float) (r2 * Math.sin(theta));
        }
        vertexpos[vertexpos.length - 5] = 0.f;
        vertexpos[vertexpos.length - 2] = 1.f;

        k = 0;

        for (int i = 0; i < quarter; i++) {
            triangles[k++] =  (i * 2 + 1);
            triangles[k++] =  (i * 2 + 3);
            triangles[k++] =  (i * 2 + 2);
            triangles[k++] =  (i * 2 + 1);
            triangles[k++] =  (i * 2 + 2);
            triangles[k++] =  (i * 2);
            triangles[k++] =  (i * 2 + 1);
            triangles[k++] =  (vertexpos.length/3-1);
            triangles[k++] =  (i * 2 + 3);
            triangles[k++] =  (i * 2);
            triangles[k++] =  (i * 2 + 2);
            triangles[k++] =  (vertexpos.length/3-2);

        }
        this.initNormals();
    }
}
