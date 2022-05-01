package fr.algo3d.model.models;

import fr.algo3d.model.Material;

/**
 * Class to represent infinite planes.
 */
public class Plane extends Model{

    /**
     * Normal.
     */
    private Vec3f normal;

    /**
     * A point of the plane.
     */
    private Vec3f A;

    /**
     * Constructor to make a plane with a normal and a point.
     * @param material
     * @param normal
     * @param A
     */
    public Plane(Material material,Vec3f normal, Vec3f A) {
        super(material);
        this.normal = normal.normalize();
        this.A = A;
    }

    /**
     * Constructor to make a plane with a normal and a distance.
     * @param material
     * @param normal
     * @param distance
     */
    public Plane(Material material,Vec3f normal, float distance){
        super(material);
        this.normal = normal.normalize();
        this.A = new Vec3f();
        this.A.setScale(distance,normal);
    }


    /**
     * Returns the closest intersection lambda.
     * @param P
     * @param v
     * @return the closest interesection lambda or -1 if there is none.
     */
    @Override
    public float getIntersection(Vec3f P, Vec3f v) {
        Vec3f tmp = new Vec3f();
        tmp.setSub(P,A);
        if (v.dotProduct(normal) != 0)
            return -tmp.dotProduct(normal)/v.dotProduct(normal);
        else return -1;
    }

    /**
     * Returns the normal at the point i.
     * @param i
     * @return
     */
    @Override
    public Vec3f getNormal(Vec3f i) {
        return normal;
    }
}
