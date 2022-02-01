package fr.univ_poitiers.dptinfo.algo3d;

public class Pipe extends Mesh {
    public Pipe(int slice, int quarter) {

        vertexpos = new float[((2) * (quarter + 1)) * 3];
        triangles = new int[quarter * 2 * 3 * 2];
        int k = 0;
        float r = 0.5f;
        for (int i = 0; i <= quarter; i++) {
            double theta = Math.toRadians((360.0 / quarter) * i);
            float x = (float) (r * Math.cos(theta));
            float z = (float) (r * Math.sin(theta));
            vertexpos[k++] = x;
            vertexpos[k++] = 0.f;
            vertexpos[k++] = z;
            vertexpos[k++] = x;
            vertexpos[k++] = 1.f;
            vertexpos[k++] = z;
        }

        k = 0;
        //faces externes
        for (int i = 0; i < quarter; i++) {
            triangles[k++] = (int) (i * 2 + 1);
            triangles[k++] = (int) (i * 2 + 2);
            triangles[k++] = (int) (i * 2 + 3);
            triangles[k++] = (int) (i * 2 + 1);
            triangles[k++] = (int) (i * 2);
            triangles[k++] = (int) (i * 2 + 2);

        }

        //faces internes
        for (int i = 0; i < quarter; i++) {
            triangles[k++] = (int) (i * 2 + 1);
            triangles[k++] = (int) (i * 2 + 3);
            triangles[k++] = (int) (i * 2 + 2);
            triangles[k++] = (int) (i * 2 + 1);
            triangles[k++] = (int) (i * 2 + 2);
            triangles[k++] = (int) (i * 2);
        }


    }
}
