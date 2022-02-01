package fr.univ_poitiers.dptinfo.algo3d;

public class Donut extends Mesh{

    public Donut(float r1, float r2, int slice, int quarter) {
        vertexpos = new float[(slice+1) * (quarter+1) *3];
        triangles= new int[slice * quarter * 2 * 3];
        int k = 0;
        for (int i = 0 ; i <= slice ; i++){
            double theta = Math.toRadians((360.0 / quarter) * i);
            for (int j = 0 ; j <= quarter ; j++){
                double phi = Math.toRadians((360.0 / quarter) * j);
                vertexpos[k++] = (float) ((r1 + r2 * Math.cos(phi))*Math.cos(theta ));
                vertexpos[k++] = (float) (r2 * Math.sin(phi));
                vertexpos[k++] = (float) ((r1 + r2 * Math.cos(phi))*Math.sin(theta ));
            }
        }

        k = 0;
        for (int i = 0 ; i < slice ; i++){
            for (int j = 0 ; j < quarter ; j++){
                triangles[k] = (int) (i * (quarter + 1) + j);
                triangles[k + 1] = (int) (i * (quarter + 1) + quarter + 2 + j);
                triangles[k + 2] = (int) (i * (quarter + 1) + 1 + j);
                triangles[k + 3] = (int) (i * (quarter + 1) + j);
                triangles[k + 4] = (int) (i * (quarter + 1) + quarter + 1 + j);
                triangles[k + 5] = (int) (i * (quarter + 1) + quarter + 2 + j);
                k += 6;
            }
        }
    }
}
