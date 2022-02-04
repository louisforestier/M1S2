package fr.univ_poitiers.dptinfo.algo3d;


/**
 * Primitive d'un plan de taille 10x10 constitué de 200 triangles
 */
public class Plane extends Mesh {

    public Plane() {
        vertexpos = new float[11*11*3];
        triangles = new int[200*3];
        int k = 0;
        for (float z = -5 ; z < 6 ; z++) {
            for (float x = -5 ; x < 6 ; x++) {
                vertexpos[k++] = x;
                vertexpos[k++] = 0.f;
                vertexpos[k++] = z;
            }
        }
        k = 0;
        for (int i = 0; i < 10 ; i++) {
            for (int j = 0; j < 10; j++) {
                triangles[k++] =  (i * (10 + 1) + j);
                triangles[k++] =  (i * (10 + 1) + 10 + 2 + j);
                triangles[k++] =  (i * (10 + 1) + 1 + j);
                triangles[k++] =  (i * (10 + 1) + j);
                triangles[k++] =  (i * (10 + 1) + 10 + 1 + j);
                triangles[k++] =  (i * (10 + 1) + 10 + 2 + j);
            }
        }
    }
}
