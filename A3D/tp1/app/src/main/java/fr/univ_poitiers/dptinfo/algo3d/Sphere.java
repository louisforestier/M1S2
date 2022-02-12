package fr.univ_poitiers.dptinfo.algo3d;


//TODO :
// - dupliquer les sommets pour donut, cylindre, pipe, pyramid et frustum
// - faire l'atténuation
// - extraire le GLSL
// - implanter le modèle blinn phong
// - implanter des ombres
// - implanter une lumière fixe
// - implanter plusieurs types de lumières
// - implanter plusieurs lumières


import android.util.Pair;

import java.util.HashMap;
import java.util.Map;

public class Sphere extends Mesh {

    private float[] vertexposIco;
    private int[] trianglesIco;
    private int nbIndicesV;
    private int nbIndicesT;
    private Map<Pair<Integer,Integer>,Integer> middleVertices = new HashMap<>();

    public Sphere(int slice, int quarter) {
        float r = 1.f;
        vertexpos = new float[((slice - 1) * (quarter + 1) + 2) * 3];
        int k = 0;
        for (int i = 1; i < slice; i++) {
            double theta = Math.toRadians(90.0 - (180.0 / slice) * i);
            for (int j = 0; j <= quarter; j++) {
                double phi = Math.toRadians((360.0 / quarter) * j);
                // formule pour l'indice sans utiliser l'astuce de la variable k, (i - 1) * (1 + quarter) * 3 + (j * 3)
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.cos(phi));
                vertexpos[k++] = (float) (r * Math.sin(theta));
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.sin(phi));
            }
        }
        vertexpos[vertexpos.length - 5] = -1;
        vertexpos[vertexpos.length - 2] = 1;

        triangles = new int[(quarter * (slice - 2) * 2 + quarter * 2) * 3];
        k = 0;
        for (int i = 0; i < slice - 2; i++) {
            for (int j = 0; j < quarter; j++) {
                triangles[k++] =  (i * (quarter + 1) + j);
                triangles[k++] =  (i * (quarter + 1) + 1 + j);
                triangles[k++] =  (i * (quarter + 1) + quarter + 2 + j);
                triangles[k++] =  (i * (quarter + 1) + j);
                triangles[k++] =  (i * (quarter + 1) + quarter + 2 + j);
                triangles[k++] =  (i * (quarter + 1) + quarter + 1 + j);
            }
        }
        for (int i = 0; i < quarter; i++) {
            triangles[k++] =  (vertexpos.length / 3 - 1);
            triangles[k++] =  (i + 1);
            triangles[k++] =  i;
        }
        for (int i = 0; i < quarter; i++) {
            triangles[k++] =  (vertexpos.length / 3 - 2);
            triangles[k++] =  (i - 1 + vertexpos.length / 3 - 2 - quarter);
            triangles[k++] =  (i + vertexpos.length / 3 - 2 - quarter);
        }
        normals = vertexpos;
    }

    public Sphere(int nbDiv) {
        vertexpos = new float[]{
                1.F, 0.F, 0.F,
                0.F, 1.F, 0.F,
                0.F, 0.F, 1.F,
                -1.F, 0.F, 0.F,
                0.F, -1.F, 0.F,
                0.F, 0.F, -1.F
        };
        nbIndicesV =  vertexpos.length;
        triangles = new int[]{
                0, 1, 2,
                0, 5, 1,
                0, 4, 5,
                0, 2, 4,
                3, 5, 4,
                3, 4, 2,
                3, 1, 5,
                3, 2, 1

        };
        nbIndicesT = 0;
        if (nbDiv > 0) {
            trianglesIco = new int[(int) (8 * 3 *  Math.pow(4,nbDiv))];
            int nbVertices = trianglesIco.length * 3 / 2 - trianglesIco.length + 6; //adapté de la relation d'euler : V - E + F = 2 => 3V - 3E + 3F = 2*3
            vertexposIco = new float[nbVertices];

            System.arraycopy(vertexpos, 0, vertexposIco, 0, vertexpos.length);
            for (int i = 0; i < triangles.length; i += 3) {
                divideTriangle(triangles[i], triangles[i + 1], triangles[i + 2], nbDiv);
            }
            vertexpos = vertexposIco;
            triangles = trianglesIco;
            normals = vertexpos;
        }

    }

    private void divideTriangle(int v1, int v2, int v3, int nbDiv) {
        if (nbDiv == 0) {
            trianglesIco[nbIndicesT] = v1;
            trianglesIco[nbIndicesT+1] = v2;
            trianglesIco[nbIndicesT+2] = v3;
            nbIndicesT+=3;
        } else {
            int middleV1V2 = getMiddle(v1,v2);
            int middleV2V3 = getMiddle(v2,v3);
            int middleV3V1 = getMiddle(v3,v1);
            divideTriangle(v1, middleV1V2,middleV3V1,nbDiv-1);
            divideTriangle(middleV1V2,v2, middleV2V3, nbDiv-1);
            divideTriangle(middleV2V3,v3,middleV3V1, nbDiv-1);
            divideTriangle(middleV1V2,middleV2V3,middleV3V1,nbDiv-1);
        }
    }

    private int getMiddle(int v1, int v2) {
        float x  = (vertexposIco[v1*3] + vertexposIco[v2*3])/2;
        float y = (vertexposIco[v1*3+1] + vertexposIco[v2*3+1])/2;
        float z = (vertexposIco[v1*3+2] + vertexposIco[v2*3+2])/2;
        double norm = Math.sqrt(x*x + y*y + z*z);
        x /= norm;
        y /= norm;
        z /= norm;
        Pair<Integer,Integer> key;
        if (v1 < v2){
            key = new Pair<>(v1,v2);
        } else {
            key = new Pair<>(v2,v1);
        }
        if (middleVertices.containsKey(key)){
            return middleVertices.get(key);
        } else {
            int vertex =  (nbIndicesV/3);
            vertexposIco[nbIndicesV] = x;
            vertexposIco[nbIndicesV + 1] = y;
            vertexposIco[nbIndicesV + 2] = z;
            nbIndicesV += 3;
            middleVertices.put(key,vertex);
            return vertex;
        }
    }


}
