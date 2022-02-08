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
        List<Integer> trianglesList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
        String lineText;
        try {
            while ((lineText = reader.readLine()) != null) {
                if (!lineText.isEmpty()) {
                    String[] data = lineText.split(" ");
                    switch (data[0]) {
                        case "#":
                            System.out.println(lineText);
                            break;
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

                        case "f":
                            trianglesList.add(Integer.parseInt((data[1].split("/"))[0]));
                            trianglesList.add(Integer.parseInt((data[2].split("/"))[0]));
                            trianglesList.add(Integer.parseInt((data[3].split("/"))[0]));
                            break;
                        default:
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        float[] vertexpos = new float[verticesList.size()];
        for (int i = 0 ; i < verticesList.size() ; i++){
            vertexpos[i] = verticesList.get(i);
        }
        int[] triangles = new int[trianglesList.size()];
        for (int i = 0 ; i < trianglesList.size(); i++){
            triangles[i] = trianglesList.get(i)-1;
        }
        float[] normals = new float[normalsList.size()];
        for (int i = 0 ; i < normalsList.size() ; i++){
            normals[i] = normalsList.get(i);
        }
        return new Mesh(vertexpos, triangles, normals);
    }

}
