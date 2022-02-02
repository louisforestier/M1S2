package fr.univ_poitiers.dptinfo.algo3d;

public class Tictac extends Mesh {

    public Tictac(int slice, int quarter) {

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
        //TODO les deux hémisphères

        k = 0;
        //faces centrales
        for (int i = 0; i < quarter; i++) {
            triangles[k++] = (int) (i * 2 + 1);
            triangles[k++] = (int) (i * 2 + 2);
            triangles[k++] = (int) (i * 2 + 3);
            triangles[k++] = (int) (i * 2 + 1);
            triangles[k++] = (int) (i * 2);
            triangles[k++] = (int) (i * 2 + 2);

        }

        vertexpos = new float[]{
                0.f,0.f,0.f,
                0.f,0.25f,1.f,
                1.f,0.25f,0.f,
                0.f,0.25f,-1.f,

                1.F, 0.F, 0.F,
                0.F, 1.F, 0.F,
                0.F, 0.F, 1.F,
                -1.F, 0.F, 0.F,
                0.F, -1.F, 0.F,
                0.F, 0.F, -1.F
        };

    }
}
