package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.GLES20;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Map;
import java.util.TreeMap;

//TODO:
// - mettre les matrices dans les classes et pas dans room
// - faire une classe vbo
// - faire un obj importer
// - faire une room avec paramètres
// - classe cube paramètre arête
// - classe pyramide paramètrée par le nombre de face
// - classe cone (juste une pyramide avec beaucoup de face)
// - classe cylinder (faire un cercle d'abord)
// - classe capsule (utiliser le cylindre et la sphère)
// - demander utilité entre plane et quad (surtout pourquoi le plane est composé de 200 triangles)
// - classe donut (torus) : dessiner un cercle autour d'un axe avec des carrés
// - joysticks


public class Sphere {

    private int glposbuffer;
    private int gltrianglesbuffer;
    private float[] vertexpos;
    private float[] vertexposIco;
    private int[] triangles;
    private int[] trianglesIco;
    private int nbIndicesV;
    private int nbIndicesT;
    private Map<Key,Integer> middleVertices = new TreeMap<>();

    public Sphere(int slice, int quarter) {
        int r = 1;
        vertexpos = new float[((slice - 1) * (quarter + 1) + 2) * 3];
        int k = 0;
        for (int i = 1; i < slice; i++) {
            double theta = Math.toRadians(90.0 - (180.0 / slice) * i);
            for (int j = 0; j <= quarter; j++) {
                double phi = Math.toRadians((360.0 / quarter) * j);
                System.out.println((i - 1) * (1 + quarter) * 3 + (j * 3) + " : theta = " + theta + "; phi = " + phi);
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.cos(phi));
                vertexpos[k++] = (float) (r * Math.cos(theta) * Math.sin(phi));
                vertexpos[k++] = (float) (r * Math.sin(theta));
            }
        }
        vertexpos[vertexpos.length - 4] = -1;
        vertexpos[vertexpos.length - 1] = 1;

        triangles = new int[(quarter * (slice - 2) * 2 + quarter * 2) * 3];
        k = 0;
        for (int i = 0; i < slice - 2; i++) {
            for (int j = 0; j < quarter; j++) {
                triangles[k] = (int) (i * (quarter + 1) + j);
                triangles[k + 1] = (int) (i * (quarter + 1) + quarter + 2 + j);
                triangles[k + 2] = (int) (i * (quarter + 1) + 1 + j);
                triangles[k + 3] = (int) (i * (quarter + 1) + j);
                triangles[k + 4] = (int) (i * (quarter + 1) + quarter + 1 + j);
                triangles[k + 5] = (int) (i * (quarter + 1) + quarter + 2 + j);
                k += 6;
            }
        }
        for (int i = 0; i < quarter; i++, k+=3) {
            triangles[k] = (int) (vertexpos.length / 3 - 1);
            triangles[k + 1] = (int) i;
            triangles[k + 2] = (int) (i + 1);
        }
        for (int i = 0; i < quarter; i++,k+=3) {
            triangles[k] = (int) (vertexpos.length / 3 - 2);
            triangles[k + 1] = (int) (i + vertexpos.length / 3 - 2 - quarter);
            triangles[k + 2] = (int) (i - 1 + vertexpos.length / 3 - 2 - quarter);
        }
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
        nbIndicesV = (int) vertexpos.length;
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
        Key key = new Key(x,y,z);
        if (middleVertices.containsKey(key)){
            return middleVertices.get(key);
        } else {
            int vertex = (int) (nbIndicesV/3);
            vertexposIco[nbIndicesV] = x;
            vertexposIco[nbIndicesV + 1] = y;
            vertexposIco[nbIndicesV + 2] = z;
            nbIndicesV += 3;
            middleVertices.put(key,vertex);
            return vertex;
        }
    }


    void initGraphics() {

        /**
         * Buffer des sommets
         */
        ByteBuffer posbytebuf = ByteBuffer.allocateDirect(vertexpos.length * Float.BYTES);
        posbytebuf.order(ByteOrder.nativeOrder());
        FloatBuffer posbuffer = posbytebuf.asFloatBuffer();
        posbuffer.put(vertexpos);
        posbuffer.position(0);


        /**
         * Buffer du des triangles
         */
        ByteBuffer trianglesbutebuf = ByteBuffer.allocateDirect(triangles.length * Integer.BYTES);
        trianglesbutebuf.order(ByteOrder.nativeOrder());
        IntBuffer trianglesbuf = trianglesbutebuf.asIntBuffer();
        trianglesbuf.put(triangles);
        trianglesbuf.position(0);

        int[] buffers = new int[1];
        GLES20.glGenBuffers(1, buffers, 0);

        glposbuffer = buffers[0];

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, vertexpos.length * Float.BYTES, posbuffer, GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);

        int[] trianglesbuffers = new int[1];
        GLES20.glGenBuffers(1, trianglesbuffers, 0);

        gltrianglesbuffer = trianglesbuffers[0];

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, triangles.length * Integer.BYTES, trianglesbuf, GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);


    }


    public void draw(final NoLightShaders shaders) {
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, triangles.length, GLES20.GL_UNSIGNED_INT, 0);

        shaders.setColor(MyGLRenderer.black);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    public void drawWithLines(final NoLightShaders shaders) {
        GLES20.glPolygonOffset(2.F, 4.F);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, triangles.length, GLES20.GL_UNSIGNED_INT, 0);

        GLES20.glDisable(GLES20.GL_POLYGON_OFFSET_FILL);
        shaders.setColor(MyGLRenderer.black);

        for (int i = 0; i < triangles.length; i += 3)
            GLES20.glDrawElements(GLES20.GL_LINE_LOOP, 3, GLES20.GL_UNSIGNED_INT, i * Integer.BYTES);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
    }



}
