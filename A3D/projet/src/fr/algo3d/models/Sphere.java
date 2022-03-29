package fr.algo3d.models;

import fr.algo3d.Material;

public class Sphere extends Model{

    private Vec3f origin;
    private float radius;

    public Sphere(Material material,Vec3f origin, float radius) {
        super(material);
        this.origin = origin;
        this.radius = radius;
    }

    @Override
    public float getIntersection(Vec3f P, Vec3f v) {
        float a = v.lengthSquare();
        Vec3f cp = new Vec3f();
        cp.setSub(P,origin);
        float b = 2 * (v.dotProduct(cp));
        float c  = (cp.lengthSquare() - radius * radius);
        float delta = b*b - 4 * a * c;
        if (delta > 0) {
            float lambda1 = (float) ((-b - Math.sqrt(delta)) / (2 * a));
            float lambda2 = (float) ((-b + Math.sqrt(delta)) / (2 * a));
            if (lambda1 > 0)
                return lambda1;
            else if (lambda2 > 0)
                return lambda2;
            else return -1;
        }else if (delta == 0){
            float lambda = -b / 2*a;
            if (lambda>0)
                return  lambda;
            else return -1;
        } else return -1;
    }

    @Override
    public Vec3f getNormal(Vec3f i) {
        Vec3f tmp = new Vec3f();
        tmp.setSub(i,origin);
        return tmp.normalize();
    }

}
