package fr.univ_poitiers.dptinfo.algo3d;

import android.opengl.GLES20;
import android.opengl.Matrix;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

public class Room {

    private int glposbuffer;
    private int glfloorbuffer;
    private int glceilingbuffer;
    private int glwallbuffer;
    private int glcontourbuffer;
    private final int[] trianglesFLoor;
    private final int[] trianglesCeiling;
    private final int[] walls;
    private int[] contours;
    private final float[] vertexpos;
    private float[] modelmatrix;
    private float[] ceilingColor;
    private float[] floorColor;
    private float[] wallColor;

    public Room(float length, float width, float height, float[] ceilingColor, float[] floorColor, float[] wallColor){
        float x = width/2;
        float z = length/2;
        vertexpos = new float[]{
                -x, 0.F, -z,    // sommet 0
                x, 0.F, -z,     // sommet 1
                x, 0.F, z,      // sommet 2
                -x, 0.F, z,     // sommet 3
                -x, height, -z,   // sommet 4
                x, height, -z,    // sommet 5
                x, height, z,     // sommet 6
                -x, height, z,    // sommet 7
                -x, height, -z,   // sommet 4
                x, height, -z,    // sommet 5
                x, 0.F, -z,     // sommet 1
                -x, 0.F, -z,    // sommet 0
                x, height, -z,    // sommet 5
                x, height, z,     // sommet 6
                x, 0.F, z,      // sommet 2
                x, 0.F, -z,     // sommet 1
                -x, height, z,    // sommet 7
                -x, height, -z,   // sommet 4
                -x, 0.F, -z,    // sommet 0
                -x, 0.F, z,     // sommet 3
                x, height, z,     // sommet 6
                -x, height, z,    // sommet 7
                -x, 0.F, z,     // sommet 3
                x, 0.F, z,      // sommet 2
                0.5F, 0.F, z,
                0.5F, height, z,
                0.5F, 2.F, z,
                -0.5F, 2.F, z,
                -0.5F, 2.5F, z,
                -0.5F, 0.F, z
        };

        trianglesFLoor = new int[]{
                0, 2, 1, //0123
                3, 2, 0
        };
        trianglesCeiling = new int[]{
                7, 5, 6,//7654
                4, 5, 7
        };
        walls = new int[]{
                8, 10, 9,//8-9-10-11
                11, 10, 8,
                12, 14, 13,//12-13-14-15
                15, 14, 12,
                16, 18, 17,//16-17-18-19
                19, 18, 16,
                //20,22,21,//20-21-22-23 ancien mur du fond
                //23,22,20
                20, 24, 25,
                23, 24, 20,
                25, 27, 28,
                26, 27, 25,
                28, 22, 21,
                29, 22, 28
        };

        contours = new int[]{
                0, 1,
                1, 2,
                2, 3,
                3, 0,
                4, 5,
                5, 6,
                6, 7,
                7, 4,
                0, 4,
                1, 5,
                2, 6,
                3, 7,
                24, 26,
                26, 27,
                27, 29
        };
        this.modelmatrix = new float[16];
        Matrix.setIdentityM(modelmatrix, 0);
        this.ceilingColor = ceilingColor;
        this.floorColor = floorColor;
        this.wallColor = wallColor;
    }

    public Room(float length, float width, float height, float[] ceilingColor, float[] floorColor, float[] wallColor, float posx, float posz, float rotation) {
        this(length, width, height, ceilingColor,floorColor, wallColor);
        Matrix.rotateM(modelmatrix, 0, rotation, 0.0F, 1.0F, 0.0F);
        Matrix.translateM(modelmatrix, 0, -posx, 0.F, -posz);
    }

    public void initGraphics(){
        /**
         * Buffer des sommets
         */
        ByteBuffer posbytebuf = ByteBuffer.allocateDirect(vertexpos.length * Float.BYTES);
        posbytebuf.order(ByteOrder.nativeOrder());
        FloatBuffer posbuffer = posbytebuf.asFloatBuffer();
        posbuffer.put(vertexpos);
        posbuffer.position(0);

        /**
         * Buffer du sol
         */
        ByteBuffer trianglesfloorbytebuf = ByteBuffer.allocateDirect(trianglesFLoor.length * Integer.BYTES);
        trianglesfloorbytebuf.order(ByteOrder.nativeOrder());
        IntBuffer trianglesfloorbuf = trianglesfloorbytebuf.asIntBuffer();
        trianglesfloorbuf.put(trianglesFLoor);
        trianglesfloorbuf.position(0);

        /**
         * Buffer du plafond
         */
        ByteBuffer trianglesceilingbytebuf = ByteBuffer.allocateDirect(trianglesCeiling.length * Integer.BYTES);
        trianglesceilingbytebuf.order(ByteOrder.nativeOrder());
        IntBuffer trianglesceilingbuf = trianglesceilingbytebuf.asIntBuffer();
        trianglesceilingbuf.put(trianglesCeiling);
        trianglesceilingbuf.position(0);

        /**
         * Buffer des murs
         */
        ByteBuffer triangleswallsbytebuf = ByteBuffer.allocateDirect(walls.length * Integer.BYTES);
        triangleswallsbytebuf.order(ByteOrder.nativeOrder());
        IntBuffer triangleswallsbuf = triangleswallsbytebuf.asIntBuffer();
        triangleswallsbuf.put(walls);
        triangleswallsbuf.position(0);

        /**
         * Buffer des contours
         */
        ByteBuffer contoursbytebuf = ByteBuffer.allocateDirect(contours.length * Integer.BYTES);
        contoursbytebuf.order(ByteOrder.nativeOrder());
        IntBuffer contoursbuf = contoursbytebuf.asIntBuffer();
        contoursbuf.put(contours);
        contoursbuf.position(0);

        int[] buffers = new int[1];
        GLES20.glGenBuffers(1,buffers,0);

        glposbuffer = buffers[0];

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
        GLES20.glBufferData(GLES20.GL_ARRAY_BUFFER, vertexpos.length * Float.BYTES, posbuffer,GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,0);

        int[] floorbuffers = new int[1];
        GLES20.glGenBuffers(1,floorbuffers,0);

        glfloorbuffer = floorbuffers[0];

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, glfloorbuffer);
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, trianglesFLoor.length * Integer.BYTES, trianglesfloorbuf,GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);

        int[] ceilingbuffers = new int[1];
        GLES20.glGenBuffers(1,ceilingbuffers,0);

        glceilingbuffer = ceilingbuffers[0];

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, glceilingbuffer);
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, trianglesCeiling.length * Integer.BYTES, trianglesceilingbuf,GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);

        int[] wallbuffers = new int[1];
        GLES20.glGenBuffers(1,wallbuffers,0);

        glwallbuffer = wallbuffers[0];

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, glwallbuffer);
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, walls.length * Integer.BYTES, triangleswallsbuf,GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);
        int[] contoursbuffers = new int[1];
        GLES20.glGenBuffers(1,contoursbuffers,0);

        glcontourbuffer = contoursbuffers[0];

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER, glcontourbuffer);
        GLES20.glBufferData(GLES20.GL_ELEMENT_ARRAY_BUFFER, contours.length * Integer.BYTES, contoursbuf,GLES20.GL_STATIC_DRAW);

        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);


        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER, glposbuffer);
    }


    public void draw(final NoLightShaders shaders){
        GLES20.glPolygonOffset(2.F,4.F);
        GLES20.glEnable(GLES20.GL_POLYGON_OFFSET_FILL);

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,glposbuffer);
        shaders.setPositionsPointer(3, GLES20.GL_FLOAT);

        shaders.setColor(ceilingColor);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,glceilingbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,trianglesCeiling.length,GLES20.GL_UNSIGNED_INT,0);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);

        shaders.setColor(floorColor);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,glfloorbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES,trianglesFLoor.length,GLES20.GL_UNSIGNED_INT,0);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);

        shaders.setColor(wallColor);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,glwallbuffer);
        GLES20.glDrawElements(GLES20.GL_TRIANGLES, walls.length,GLES20.GL_UNSIGNED_INT,0);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);

        GLES20.glDisable(GLES20.GL_POLYGON_OFFSET_FILL);

        shaders.setColor(MyGLRenderer.black);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,glcontourbuffer);
        GLES20.glDrawElements(GLES20.GL_LINES, contours.length,GLES20.GL_UNSIGNED_INT, 0);
        GLES20.glBindBuffer(GLES20.GL_ELEMENT_ARRAY_BUFFER,0);

        GLES20.glBindBuffer(GLES20.GL_ARRAY_BUFFER,0);

        MyGLRenderer.checkGlError("glDrawArrays (GL_TRIANGLES)");
    }

    public void draw(final NoLightShaders shaders, float[] viewmatrix)
    {
        float[] modelviewmatrixroom = new float[16];
        Matrix.multiplyMM(modelviewmatrixroom,0,viewmatrix, 0, modelmatrix,0);
        shaders.setModelViewMatrix(modelviewmatrixroom);
        this.draw(shaders);
    }

}

