package fr.univ_poitiers.dptinfo.algo3d.mesh;

import fr.univ_poitiers.dptinfo.algo3d.Vec3f;

public class Pyramid extends Mesh {

    public Pyramid(int quarter) {
        vertexpos = new float[(2*(quarter+1)+1+quarter)*3]; //+quarter pour le sommet,+1 pour la base et +1 pour le sommet de jointure de la base (le sommet répété pour theta = 0 et theta = 360)
        triangles = new int[quarter*2*3];
        int k = 0;
        float r = 1.f;
        float height = 1.f;
        for (int i = 0 ; i <= quarter ; i++){
            double theta = Math.toRadians((360.0 / quarter) * i);
            vertexpos[k++] = (float) (r  * Math.cos(theta));
            vertexpos[k++] = 0.f;
            vertexpos[k++] = (float) (r * Math.sin(theta));
        }
        for (int i = 0 ; i <= quarter ; i++){
            double theta = Math.toRadians((360.0 / quarter) * i);
            vertexpos[k++] = (float) (r  * Math.cos(theta));
            vertexpos[k++] = 0.f;
            vertexpos[k++] = (float) (r * Math.sin(theta));
        }
        for (int i = 0 ; i < quarter ; i++){
            vertexpos[k++] = 0.f;
            vertexpos[k++] = height;
            vertexpos[k++] = 0.f;
        }

        vertexpos[k++] = 0.f;
        vertexpos[k++] = 0.f;
        vertexpos[k++] = 0.f;

        k=0;

        for (int i = 0 ; i < quarter ; i++){
            triangles[k++] = i;
            triangles[k++] = (quarter+1)*2+i;
            triangles[k++] = i+1;
            triangles[k++] = i+quarter+1;
            triangles[k++] = i+quarter+1+1;
            triangles[k++] =(vertexpos.length-1)/3;
        }
        k=0;

        Vec3f p1 = new Vec3f(vertexpos[triangles[0]*3],vertexpos[triangles[0]*3+1],vertexpos[triangles[0]*3+2]);
        Vec3f p2 = new Vec3f(vertexpos[triangles[0+1]*3],vertexpos[triangles[0+1]*3+1],vertexpos[triangles[0+1]*3+2]);
        Vec3f p3 = new Vec3f(vertexpos[triangles[0+2]*3],vertexpos[triangles[0+2]*3+1],vertexpos[triangles[0+2]*3+2]);
        Vec3f n = getNormal(p1,p2,p3);

        normals = new float[vertexpos.length];
//TODO:calculer les x et z des normals des sommets et mettre une garde sur les rayons du tornc qui doivent etre sup a 0
        for (int i = 0 ; i <= quarter ; i++){
            double theta = Math.toRadians((360.0 / quarter) * i);
            normals[k++] = (float) (r  * Math.cos(theta));
            normals[k++] = n.y;
            normals[k++] = (float) (r * Math.sin(theta));
        }
        for (int i = 0 ; i <= quarter ; i++){
            normals[k++] = 0.f;
            normals[k++] = -1.f;
            normals[k++] = 0.f;
        }
        for (int i = 0 ; i < quarter ; i++){
            double theta1 = Math.toRadians((360.0 / quarter) * i);
            double theta2 = Math.toRadians((360.0 / quarter) * (i+1));
            float x  = (float) (((r  * Math.cos(theta1)) + (r  * Math.cos(theta2)))/2);
            float z  = (float) (((r  * Math.sin(theta1)) + (r  * Math.sin(theta2)))/2);
            double norm = Math.sqrt(x*x + n.y*n.y + z*z);
            x /= norm;
            z /= norm;

            normals[k++] = x;
            normals[k++] = (float) (n.y/norm);
            normals[k++] = z;
        }

        normals[k++] = 0.f;
        normals[k++] = -1.f;
        normals[k++] = 0.f;
        this.calculateFlatShadingNormals();
    }
}
