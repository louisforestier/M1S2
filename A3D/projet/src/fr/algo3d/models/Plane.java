package fr.algo3d.models;

import fr.algo3d.Material;

public class Plane extends Model{

    private Vec3f normal;
    private Vec3f A;

    public Plane(Material material,Vec3f normal, Vec3f A) {
        super(material);
        this.normal = normal;
        this.A = A;
    }

    public Plane(Material material,Vec3f normal, float distance){
        super(material);
        this.normal = normal;
        this.A = normal.scale(distance);
    }

    public Plane(Material material) {
        super(material);
    }

    @Override
    public float getIntersection(Vec3f P, Vec3f v) {
        Vec3f tmp = new Vec3f();
        tmp.setSub(P,A);
        if (v.dotProduct(normal) != 0)
            return -tmp.dotProduct(normal)/v.dotProduct(normal);
        else return -1;
    }

    @Override
    public Vec3f getNormal(Vec3f i) {
        return normal;
    }
}
