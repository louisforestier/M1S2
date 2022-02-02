package fr.univ_poitiers.dptinfo.algo3d;

public class Pyramid extends Mesh {

    public Pyramid(int quarter) {
        vertexpos = new float[(quarter+3)*3]; //+1 pour le sommet,+1 pour la base et +1 pour le sommet de jointure de la base (le sommet répété pour theta = 0 et theta = 360)
        triangles = new int[quarter*2*3];
        int k = 0;
        float r = 1.f;
        float height = 1.f;
        vertexpos[k++] = 0.f;
        vertexpos[k++] = 0.f;
        vertexpos[k++] = 0.f;
        for (int i = 0 ; i <= quarter ; i++){
            double theta = Math.toRadians((360.0 / quarter) * i);
            vertexpos[k++] = (float) (r  * Math.cos(theta));
            vertexpos[k++] = 0.f;
            vertexpos[k++] = (float) (r * Math.sin(theta));
        }
        vertexpos[k++] = 0.f;
        vertexpos[k++] = height;
        vertexpos[k++] = 0.f;

        k=0;

        for (int i = 1 ; i <= quarter ; i++){
            triangles[k++] = i;
            triangles[k++] = i+1;
            triangles[k++] = (vertexpos.length-1)/3;
            triangles[k++] = i+1;
            triangles[k++] = i;
            triangles[k++] = 0;
        }
    }
}
