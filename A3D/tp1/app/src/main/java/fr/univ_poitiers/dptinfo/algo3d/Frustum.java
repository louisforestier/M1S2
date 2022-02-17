package fr.univ_poitiers.dptinfo.algo3d;

public class Frustum extends Mesh {

    public Frustum(float r1, float r2, int quarter) {
        vertexpos = new float[((2*2) * (quarter + 1)+2) * 3];
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
            triangles[k++] =  ((i+quarter+1) * 2 + 1);
            triangles[k++] =  (vertexpos.length/3-1);
            triangles[k++] =  ((i+quarter+1) * 2 + 3);
            triangles[k++] =  ((i+quarter+1) * 2);
            triangles[k++] =  ((i+quarter+1) * 2 + 2);
            triangles[k++] =  (vertexpos.length/3-2);
        }
        k=0;

        Vec3f p1 = new Vec3f(vertexpos[triangles[0]*3],vertexpos[triangles[0]*3+1],vertexpos[triangles[0]*3+2]);
        Vec3f p2 = new Vec3f(vertexpos[triangles[0+1]*3],vertexpos[triangles[0+1]*3+1],vertexpos[triangles[0+1]*3+2]);
        Vec3f p3 = new Vec3f(vertexpos[triangles[0+2]*3],vertexpos[triangles[0+2]*3+1],vertexpos[triangles[0+2]*3+2]);
        Vec3f n = getNormal(p1,p2,p3);

        normals = new float[vertexpos.length];

        for (int i = 0; i <= quarter; i++) {
            double theta = Math.toRadians((360.0 / quarter) * i);
            normals[k++] = (float) (r1 * Math.cos(theta));
            normals[k++] = n.y;
            normals[k++] = (float) (r1 * Math.sin(theta));
            normals[k++] = (float) (r2 * Math.cos(theta));
            normals[k++] = n.y;
            normals[k++] = (float) (r2 * Math.sin(theta));
        }
        for (int i = 0; i <= quarter; i++) {
            normals[k++] = 0.f;
            normals[k++] = -1.f;
            normals[k++] = 0.f;
            normals[k++] = 0.f;
            normals[k++] = 1.f;
            normals[k++] = 0.f;
        }
        //pas besoin d'initialiser les autres parties de ces sommets car java initialise déjà ces parties à 0, donc je ne modifie que la composante en y
        normals[normals.length - 5] = -1.f;
        normals[normals.length - 2] = 1.f;

    }
}
