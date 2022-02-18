package fr.univ_poitiers.dptinfo.algo3d;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class OBJImporter {

    public static Mesh importOBJ(InputStream stream) {
        List<Float> verticesList = new ArrayList<>();
        List<Float> normalsList = new ArrayList<>();
        List<Float> texturesList = new ArrayList<>();
        List<OBJFace> trianglesList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String lineText;
        try {
            while ((lineText = reader.readLine()) != null) {
                if (!lineText.isEmpty()) {
                    String[] data = lineText.split(" ");
                    switch (data[0]) {
                        case "v":
                            verticesList.add(Float.parseFloat(data[1]));
                            verticesList.add(Float.parseFloat(data[2]));
                            verticesList.add(Float.parseFloat(data[3]));
                            break;
                        case "vn":
                            normalsList.add(Float.parseFloat(data[1]));
                            normalsList.add(Float.parseFloat(data[2]));
                            normalsList.add(Float.parseFloat(data[3]));
                            break;
                        case "vt":
                            texturesList.add(Float.parseFloat(data[1]));
                            texturesList.add(Float.parseFloat(data[2]));
                            break;

                        case "f":
                            trianglesList.add(new OBJFace(data));
                            break;
                        default:
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        Mesh mesh;
        float[] vertexpos = new float[trianglesList.size()*9];
        int[] triangles = new int[trianglesList.size()*3];
        if (normalsList.isEmpty() && texturesList.isEmpty()) {
            for (int i = 0; i < verticesList.size(); i++) {
                vertexpos[i] = verticesList.get(i);
            }
            for (int i = 0; i < trianglesList.size(); i++) {
                triangles[i*3] = trianglesList.get(i).getV1().getV() - 1;
                triangles[i*3+1] = trianglesList.get(i).getV2().getV() - 1;
                triangles[i*3+2] = trianglesList.get(i).getV3().getV() - 1;
            }
            mesh = new Mesh(vertexpos, triangles);
            mesh.initNormals();
        } else if (texturesList.isEmpty()){
            float[] normals = new float[vertexpos.length];
            for (int i = 0 ; i < trianglesList.size() ; i++){
                vertexpos[i*9] = verticesList.get((trianglesList.get(i).getV1().getV()-1)*3);
                vertexpos[i*9+1] = verticesList.get((trianglesList.get(i).getV1().getV()-1)*3+1);
                vertexpos[i*9+2] = verticesList.get((trianglesList.get(i).getV1().getV()-1)*3+2);
                vertexpos[i*9+3] = verticesList.get((trianglesList.get(i).getV2().getV()-1)*3);
                vertexpos[i*9+4] = verticesList.get((trianglesList.get(i).getV2().getV()-1)*3+1);
                vertexpos[i*9+5] = verticesList.get((trianglesList.get(i).getV2().getV()-1)*3+2);
                vertexpos[i*9+6] = verticesList.get((trianglesList.get(i).getV3().getV()-1)*3);
                vertexpos[i*9+7] = verticesList.get((trianglesList.get(i).getV3().getV()-1)*3+1);
                vertexpos[i*9+8] = verticesList.get((trianglesList.get(i).getV3().getV()-1)*3+2);
                normals[i*9] = verticesList.get((trianglesList.get(i).getV1().getVn()-1)*3);
                normals[i*9+1] = verticesList.get((trianglesList.get(i).getV1().getVn()-1)*3+1);
                normals[i*9+2] = verticesList.get((trianglesList.get(i).getV1().getVn()-1)*3+2);
                normals[i*9+3] = verticesList.get((trianglesList.get(i).getV2().getVn()-1)*3);
                normals[i*9+4] = verticesList.get((trianglesList.get(i).getV2().getVn()-1)*3+1);
                normals[i*9+5] = verticesList.get((trianglesList.get(i).getV2().getVn()-1)*3+2);
                normals[i*9+6] = verticesList.get((trianglesList.get(i).getV3().getVn()-1)*3);
                normals[i*9+7] = verticesList.get((trianglesList.get(i).getV3().getVn()-1)*3+1);
                normals[i*9+8] = verticesList.get((trianglesList.get(i).getV3().getVn()-1)*3+2);
                triangles[i*3] = i*3;
                triangles[i*3+1] = i*3+1;
                triangles[i*3+2] = i*3+2;
            }
            mesh = new Mesh(vertexpos,triangles,normals);
        } else {
            float[] normals = new float[vertexpos.length];
            float[] textures = new float[vertexpos.length];
            for (int i = 0 ; i < trianglesList.size() ; i++){
                vertexpos[i*9] = verticesList.get((trianglesList.get(i).getV1().getV()-1)*3);
                vertexpos[i*9+1] = verticesList.get((trianglesList.get(i).getV1().getV()-1)*3+1);
                vertexpos[i*9+2] = verticesList.get((trianglesList.get(i).getV1().getV()-1)*3+2);
                vertexpos[i*9+3] = verticesList.get((trianglesList.get(i).getV2().getV()-1)*3);
                vertexpos[i*9+4] = verticesList.get((trianglesList.get(i).getV2().getV()-1)*3+1);
                vertexpos[i*9+5] = verticesList.get((trianglesList.get(i).getV2().getV()-1)*3+2);
                vertexpos[i*9+6] = verticesList.get((trianglesList.get(i).getV3().getV()-1)*3);
                vertexpos[i*9+7] = verticesList.get((trianglesList.get(i).getV3().getV()-1)*3+1);
                vertexpos[i*9+8] = verticesList.get((trianglesList.get(i).getV3().getV()-1)*3+2);
                textures[i*9] = verticesList.get((trianglesList.get(i).getV1().getVt()-1)*3);
                textures[i*9+1] = verticesList.get((trianglesList.get(i).getV1().getVt()-1)*3+1);
                textures[i*9+2] = verticesList.get((trianglesList.get(i).getV1().getVt()-1)*3+2);
                textures[i*9+3] = verticesList.get((trianglesList.get(i).getV2().getVt()-1)*3);
                textures[i*9+4] = verticesList.get((trianglesList.get(i).getV2().getVt()-1)*3+1);
                textures[i*9+5] = verticesList.get((trianglesList.get(i).getV2().getVt()-1)*3+2);
                textures[i*9+6] = verticesList.get((trianglesList.get(i).getV3().getVt()-1)*3);
                textures[i*9+7] = verticesList.get((trianglesList.get(i).getV3().getVt()-1)*3+1);
                textures[i*9+8] = verticesList.get((trianglesList.get(i).getV3().getVt()-1)*3+2);
                normals[i*9] = verticesList.get((trianglesList.get(i).getV1().getVn()-1)*3);
                normals[i*9+1] = verticesList.get((trianglesList.get(i).getV1().getVn()-1)*3+1);
                normals[i*9+2] = verticesList.get((trianglesList.get(i).getV1().getVn()-1)*3+2);
                normals[i*9+3] = verticesList.get((trianglesList.get(i).getV2().getVn()-1)*3);
                normals[i*9+4] = verticesList.get((trianglesList.get(i).getV2().getVn()-1)*3+1);
                normals[i*9+5] = verticesList.get((trianglesList.get(i).getV2().getVn()-1)*3+2);
                normals[i*9+6] = verticesList.get((trianglesList.get(i).getV3().getVn()-1)*3);
                normals[i*9+7] = verticesList.get((trianglesList.get(i).getV3().getVn()-1)*3+1);
                normals[i*9+8] = verticesList.get((trianglesList.get(i).getV3().getVn()-1)*3+2);
                triangles[i*3] = i*3;
                triangles[i*3+1] = i*3+1;
                triangles[i*3+2] = i*3+2;
            }
            mesh = new Mesh(vertexpos,triangles,normals,textures);
        }
        return mesh;
    }

}
