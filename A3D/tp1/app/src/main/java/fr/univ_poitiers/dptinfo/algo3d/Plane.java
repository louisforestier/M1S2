package fr.univ_poitiers.dptinfo.algo3d;


/**
 * Primitive d'un plan de taille 10x10 constitué de 200 triangles
 */
public class Plane extends Mesh {

    public static final Plane INSTANCE = new Plane();


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


        normals = new float[11*11*3];
        for (int i = 0 ; i < normals.length ; i+=3) {
            normals[i] = 1.f;
            normals[i+1] = 0.f;
            normals[i+2] = 0.f;
        }
/*
        for (int i = 0 ; i < triangles.length ; i+=3) {
            Vec3f p1 = new Vec3f(vertexpos[triangles[i]],vertexpos[triangles[i]+1],vertexpos[triangles[i]+2]);
            Vec3f p2 = new Vec3f(vertexpos[triangles[i+1]],vertexpos[triangles[i+1]+1],vertexpos[triangles[i+1]+2]);
            Vec3f p3 = new Vec3f(vertexpos[triangles[i+2]],vertexpos[triangles[i+2]+1],vertexpos[triangles[i+2]+2]);
            Vec3f n1 = getNormal(p1,p2,p3);
            Vec3f n2 = getNormal(p2,p1,p3);
            Vec3f n3 = getNormal(p3,p1,p2);
            normals[triangles[i]] = n1.x;
            normals[triangles[i]+1] = n1.y;
            normals[triangles[i]+2] = n1.z;
            normals[triangles[i+1]] = n2.x;
            normals[triangles[i+1]+1] = n2.y;
            normals[triangles[i+1]+2] = n2.z;
            normals[triangles[i+2]] = n3.x;
            normals[triangles[i+2]+1] = n3.y;
            normals[triangles[i+2]+2] = n3.z;
        }
*/

    }

    Vec3f getNormal(Vec3f p1, Vec3f p2, Vec3f p3){
        Vec3f v1 = p2.sub(p1);
        Vec3f v2 = p3.sub(p1);
        Vec3f n1 = new Vec3f();
        n1.setCrossProduct(v1,v2);
        n1.normalize();
        return n1;
    }
}
