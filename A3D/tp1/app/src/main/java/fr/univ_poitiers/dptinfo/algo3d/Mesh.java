package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.GLES20;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class Mesh {
    private int glposbuffer;
    private int gltrianglesbuffer;
    private int glnormalbuffer;
    protected float[] vertexpos;
    protected int[] triangles;
    protected float[] normals;

    public Mesh() {
        this.normals = new float[]{0.f, 0.f, 0.f};
    }


    public Mesh(float[] vertexpos, int[] triangles) {
        this.vertexpos = vertexpos;
        this.triangles = triangles;
        this.normals = new float[]{0.f, 0.f, 0.f};
    }

    public Mesh(float[] vertexpos, int[] triangles, float[] normals) {
        this.vertexpos = vertexpos;
        this.triangles = triangles;
        this.normals = normals;
    }

    public Mesh(float[] vertexpos, int[] triangles, float[] normals, float[] textures) {
        this.vertexpos = vertexpos;
        this.triangles = triangles;
        this.normals = normals;
    }

    public void calculateFlatShadingNormals() {
        normals = new float[vertexpos.length];
        for (int i = 0; i < triangles.length; i += 3) {
            Vec3f p1 = new Vec3f(vertexpos[triangles[i] * 3], vertexpos[triangles[i] * 3 + 1], vertexpos[triangles[i] * 3 + 2]);
            Vec3f p2 = new Vec3f(vertexpos[triangles[i + 1] * 3], vertexpos[triangles[i + 1] * 3 + 1], vertexpos[triangles[i + 1] * 3 + 2]);
            Vec3f p3 = new Vec3f(vertexpos[triangles[i + 2] * 3], vertexpos[triangles[i + 2] * 3 + 1], vertexpos[triangles[i + 2] * 3 + 2]);
            Vec3f n = getNormal(p1, p2, p3);
            normals[triangles[i] * 3] = n.x;
            normals[triangles[i] * 3 + 1] = n.y;
            normals[triangles[i] * 3 + 2] = n.z;
            normals[triangles[i + 1] * 3] = n.x;
            normals[triangles[i + 1] * 3 + 1] = n.y;
            normals[triangles[i + 1] * 3 + 2] = n.z;
            normals[triangles[i + 2] * 3] = n.x;
            normals[triangles[i + 2] * 3 + 1] = n.y;
            normals[triangles[i + 2] * 3 + 2] = n.z;
        }
    }

    public void calculateSmoothShadingNormals() {
        normals = new float[vertexpos.length];
        for (int i = 0; i < triangles.length; i += 3) {
            Vec3f p1 = new Vec3f(vertexpos[triangles[i] * 3], vertexpos[triangles[i] * 3 + 1], vertexpos[triangles[i] * 3 + 2]);
            Vec3f p2 = new Vec3f(vertexpos[triangles[i + 1] * 3], vertexpos[triangles[i + 1] * 3 + 1], vertexpos[triangles[i + 1] * 3 + 2]);
            Vec3f p3 = new Vec3f(vertexpos[triangles[i + 2] * 3], vertexpos[triangles[i + 2] * 3 + 1], vertexpos[triangles[i + 2] * 3 + 2]);
            Vec3f v1 = new Vec3f();
            v1.setSub(p3, p1);
            Vec3f v2 = new Vec3f();
            v2.setSub(p3, p2);
            Vec3f n = new Vec3f();
            n.setCrossProduct(v1, v2);
            float a1 = calcAngle(p1,p2,p3);
            float a2 = calcAngle(p2,p3,p1);
            float a3 = calcAngle(p3,p1,p2);
            Vec3f n1 = n.scale(a1);
            Vec3f n2 = n.scale(a2);
            Vec3f n3 = n.scale(a3);
            normals[triangles[i] * 3] += n1.x;
            normals[triangles[i] * 3 + 1] += n1.y;
            normals[triangles[i] * 3 + 2] += n1.z;
            normals[triangles[i + 1] * 3] += n2.x;
            normals[triangles[i + 1] * 3 + 1] += n2.y;
            normals[triangles[i + 1] * 3 + 2] += n2.z;
            normals[triangles[i + 2] * 3] += n3.x;
            normals[triangles[i + 2] * 3 + 1] += n3.y;
            normals[triangles[i + 2] * 3 + 2] += n3.z;
        }
        for (int i = 0 ; i < normals.length ; i+=3) {
            Vec3f n = new Vec3f(normals[i],normals[i+1],normals[i+2]);
            n.normalize();
            normals[i] = n.x;
            normals[i+1] = n.y;
            normals[i+2] = n.z;
        }
    }

    float calcAngle(Vec3f p1, Vec3f p2, Vec3f p3){
        Vec3f v1 = new Vec3f();
        v1.setSub(p2, p1);
        Vec3f v2 = new Vec3f();
        v2.setSub(p3, p1);
        float angle = (float) Math.acos(v1.dotProduct(v2) /(v1.length()* v2.length()));
        return angle;
    }

    Vec3f getNormal(Vec3f p1, Vec3f p2, Vec3f p3) {
        Vec3f v1 = new Vec3f();
        v1.setSub(p2, p1);
        Vec3f v2 = new Vec3f();
        v2.setSub(p3, p1);
        Vec3f n = new Vec3f();
        n.setCrossProduct(v1, v2);
        n.normalize();
        return n;
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
         * Buffer des triangles
         */
        ByteBuffer trianglesbutebuf = ByteBuffer.allocateDirect(triangles.length * Integer.BYTES);
        trianglesbutebuf.order(ByteOrder.nativeOrder());
        IntBuffer trianglesbuf = trianglesbutebuf.asIntBuffer();
        trianglesbuf.put(triangles);
        trianglesbuf.position(0);

        /**
         *
         * Buffer des normals
         */
        Log.i("INFO", "buffer des normals");
        ByteBuffer normalbytebuf = ByteBuffer.allocateDirect(normals.length * Float.BYTES);
        normalbytebuf.order(ByteOrder.nativeOrder());
        FloatBuffer normalbuffer = normalbytebuf.asFloatBuffer();
        normalbuffer.put(normals);
        normalbuffer.position(0);


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

        int[] normalsbuffers = new int[1];
        GLES20.glGenBuffers(1, normalsbuffers, 0);

        glnormalbuffer = normalsbuffers[0];

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glnormalbuffer);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, normals.length * Float.BYTES, normalbuffer, GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);
        Log.i("INFO", "fin init graphics");
    }

    public void draw(final LightingShaders shaders) {

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glnormalbuffer);
        shaders.setNormalsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, triangles.length, GLES20.GL_UNSIGNED_INT, 0);


        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);


    }

    public void drawWithLines(final LightingShaders shaders) {
        GLES20.glPolygonOffset(2.F, 4.F);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glnormalbuffer);
        shaders.setNormalsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, triangles.length, GLES20.GL_UNSIGNED_INT, 0);

        GLES20.glDisable(GLES20.GL_POLYGON_OFFSET_FILL);
        shaders.setMaterialColor(MyGLRenderer.black);

        for (int i = 0; i < triangles.length; i += 3)
            GLES20.glDrawElements(GLES20.GL_LINE_LOOP, 3, GLES20.GL_UNSIGNED_INT, i * Integer.BYTES);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);


    }

    public void drawLinesOnly(final LightingShaders shaders) {

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glnormalbuffer);
        shaders.setNormalsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);

        shaders.setMaterialColor(MyGLRenderer.black);

        for (int i = 0; i < triangles.length; i += 3)
            GLES20.glDrawElements(GLES20.GL_LINE_LOOP, 3, GLES20.GL_UNSIGNED_INT, i * Integer.BYTES);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);
        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, 0);

    }

    public void draw(LightingShaders shaders, DrawMode drawMode) {
        switch (drawMode) {
            case TRIANGLES:
                draw(shaders);
                break;
            case WIREFRAME:
                drawLinesOnly(shaders);
                break;
            case TRIANGLES_AND_WIREFRAME:
                drawWithLines(shaders);
        }
    }
}
