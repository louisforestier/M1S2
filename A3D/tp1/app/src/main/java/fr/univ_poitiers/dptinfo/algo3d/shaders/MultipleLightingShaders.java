package fr.univ_poitiers.dptinfo.algo3d.shaders;

import android.content.Context;
import android.opengl.GLES20;

import fr.univ_poitiers.dptinfo.algo3d.MainActivity;

/**
 * Abstract class to represent shaders (vertex and fragment ones) that allow a
 * computation of plong lighting with no texture.
 * @author Philippe Meseure
 * @version 1.0
 */
public abstract class MultipleLightingShaders extends BasicShaders
{
    // ==============================
    // Uniform variables for matrices
    // ==============================
    /**
     * GLSL uniform transformation matrix for normal (from object's space to viewer's space)
     */
    protected int uNormalMatrix;
    // ======================================
    // Uniform variables for (a single) light
    // ======================================
    /**
     * GLSL uniform boolean value to turn lighting on/off
     */
    protected int uLighting;
    // =====================================
    // Uniform variables for object material
    // =====================================
    /**
     * GLSL uniform boolean to normalize normals (or not)
     */
    protected int uNormalizing;
    /**
     * GLSL uniform material color of the object
     */
    protected int uMaterialColor;
    /**
     * GLSL uniform material specular color
     */
    protected int uMaterialSpecular;
    /**
     * GLSL uniform Shininess of the material (for specular component)
     */
    protected int uMaterialShininess;
    // ================================
    // Attributes to manage GLES arrays
    // ================================
    /**
     * GLSL attribute for vertex normal arrays
     */
    protected int aVertexNormal;


    /**
     * Constructor. nothing to do, everything is done in the super class...
     */
    public MultipleLightingShaders(Context context)
    {
        super(context);
    }


    /**
     * Get all the uniform variables and attributes defined in the shaders
     */
    @Override
    public void findVariables()
    {
        super.findVariables();
        // Variables for matrices
        this.uNormalMatrix = GLES20.glGetUniformLocation(this.shaderprogram, "uNormalMatrix");
        if (this.uNormalMatrix==-1) throw new RuntimeException("uNormalMatrix not found in shaders");

        // Variables for light source
        this.uLighting = GLES20.glGetUniformLocation(this.shaderprogram, "uLighting");
        if (this.uLighting==-1) MainActivity.log("uLighting not found in shaders...");


        // Variables for material
        this.uNormalizing = GLES20.glGetUniformLocation(this.shaderprogram, "uNormalizing");
        if (this.uNormalizing==-1) MainActivity.log("uNormalizing not found in shaders...");

        this.uMaterialColor = GLES20.glGetUniformLocation(this.shaderprogram, "uMaterialColor");
        if (this.uMaterialColor==-1) throw new RuntimeException("uMaterialColor not found in shaders");

        this.uMaterialSpecular = GLES20.glGetUniformLocation(this.shaderprogram, "uMaterialSpecular");
        if (this.uMaterialSpecular==-1) MainActivity.log("Warning: uMaterialSpecular not found in shaders");

        this.uMaterialShininess = GLES20.glGetUniformLocation(this.shaderprogram, "uMaterialShininess");
        if (this.uMaterialShininess==-1) MainActivity.log("Warning: uMaterialShininess not found in shaders");

        // vertex attributes
        this.aVertexNormal = GLES20.glGetAttribLocation(this.shaderprogram, "aVertexNormal");
        if (this.aVertexNormal==-1) throw new RuntimeException("aVertexNormal not found in shaders");
        GLES20.glEnableVertexAttribArray(this.aVertexNormal);
    }

    /**
     * Convert a ModelView Matrix into a NormalMatrix
     * More exactly, the translation and scale component of the transformation must be ignored,
     * while only the rotation component must be taken into account
     * OpenGL redbook mainly recommend a transposition of the inverse matrix...
     * @param a ModelView matrix (in)
     * @param b Normal matrix (out)
     */
    static void convertMVtoNM(final float[] a,float[] b)
    {
        float c=a[0],d=a[1],e=a[2],
          g=a[4],f=a[5],h=a[6],
          i=a[8],j=a[9],k=a[10],
          l=k*f-h*j,o=-k*g+h*i,m=j*g-f*i,n=c*l+d*o+e*m;
        if(n==0.) return;
        n=1.f/n;
        b[0]=l*n; b[3]=(-k*d+e*j)*n; b[6]=(h*d-e*f)*n;
        b[1]=o*n; b[4]=(k*c-e*i)*n; b[7]=(-h*c+e*g)*n;
        b[2]=m*n; b[5]=(-j*c+d*i)*n; b[8]=(f*c-d*g)*n;
    }

    // ================
    // Matrix functions
    // ================
    /**
     * Set the model view matrix (transformation from object's space to viewer's space)
     * @param matrix matrix to set to the GLSL modelview matrix
     */
    @Override
    public void setModelViewMatrix(final float[] matrix)
    {
        float[] normal_matrix=new float[9];

        // Set modelview matrix
        super.setModelViewMatrix(matrix);

        // Set normal matrix according to the modelview matrix.
        // Scaling and translation must not be applied, only rotations...
        convertMVtoNM(matrix, normal_matrix);
        GLES20.glUniformMatrix3fv(this.uNormalMatrix, 1, false, normal_matrix,0);
    }

    // =====================
    // Source light function
    // =====================

    /**
     * Set lighting on/off
     * @param state on/off value
     */
    public void setLighting(final boolean state)
    {
        if (this.uLighting!=-1) GLES20.glUniform1i(this.uLighting,state?1:0);
    }


    // ==================
    // Material functions
    // ==================

    /**
     * Set normalizing of normals
     * @param state on/off
     */
    public void setNormalizing(final boolean state)
    {
        if (this.uNormalizing!=-1)
            GLES20.glUniform1i(this.uNormalizing,state?1:0);
    }

    /**
     * Set object (diffuse and ambient) color
     * @param matcolor color to set to the object
     */
    public void setMaterialColor(final float[] matcolor)
    {
        GLES20.glUniform4fv(this.uMaterialColor,1,matcolor,0);
    }

    /**
     * Set object specular color
     * @param matspec specular color to set to the object
     */
    public void setMaterialSpecular(final float[] matspec)
    {
        GLES20.glUniform4fv(this.uMaterialSpecular,1,matspec,0);
    }

    /**
     * Set the object shininess (for specular component)
     * @param shininess shininess of the object
     */
    public void setMaterialShininess(final float shininess)
    {
        GLES20.glUniform1f(this.uMaterialShininess,shininess);
    }

    // ===================
    // Attributes handling
    // ===================
    /**
     * Set normal array for future drawings
     * @param size number of coordinates by normals
     * @param dtype type of coordinates
     */
    public void setNormalsPointer(int size,int dtype)
    {
        GLES20.glVertexAttribPointer(this.aVertexNormal, size, dtype, false, 0, 0);
    }


}
