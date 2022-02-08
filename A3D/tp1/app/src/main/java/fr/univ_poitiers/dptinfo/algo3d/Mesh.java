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

    public Mesh(){
        this.normals = new float[]{0.f,0.f,0.f};
    }


    public Mesh(float[] vertexpos, int[] triangles) {
        this.vertexpos = vertexpos;
        this.triangles = triangles;
        this.normals = new float[]{0.f,0.f,0.f};
    }
    public Mesh(float[] vertexpos, int[] triangles, float[] normals) {
        this.vertexpos = vertexpos;
        this.triangles = triangles;
        this.normals = normals;
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
        Log.i("INFO","buffer des normals");
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
        Log.i("INFO","fin init graphics");
    }

    public void draw(final LightingShaders shaders) {

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, triangles.length, GLES20.GL_UNSIGNED_INT, 0);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, glnormalbuffer);
        shaders.setNormalsPointer(3,GLES20.GL_FLOAT);

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
    public void drawLinesOnly(final NoLightShaders shaders) {

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, gltrianglesbuffer);

        shaders.setColor(MyGLRenderer.black);

        for (int i = 0; i < triangles.length; i += 3)
            GLES20.glDrawElements(GLES20.GL_LINE_LOOP, 3, GLES20.GL_UNSIGNED_INT, i * Integer.BYTES);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, 0);

    }

}
