package fr.algo3d;

import fr.algo3d.Material;
import fr.algo3d.lights.Light;
import fr.algo3d.models.*;

import java.util.ArrayList;
import java.util.List;

public class Scene {

    private List<Model> models = new ArrayList<>();
    private List<Light> lights = new ArrayList<>();

    public Scene() {
        Material green = new Material(Color.green,Color.white,32);
        Material orange = new Material(Color.orange,Color.white,100,0.f,0,1);
        Material cyan = new Material(Color.cyan,Color.white,32);
        models.add(new Plane(green,new Vec3f(0,1,0),-2));
        models.add(new Plane(cyan,new Vec3f(-1,1,1),-20));
        models.add(new Sphere(orange,new Vec3f(0,0,-7.5f),2));
        lights.add(new Light(new Vec3f(-1,1,0), Color.darkgray,Color.lightgray,Color.white));
    }

    Color findColor(Vec3f P, Vec3f v, int depth){
        Color color = new Color();
        //List<Vec3f> positions = lights.stream().map(light -> light.getPosition()).collect(Collectors.toList());
        //models.parallelStream().collect(Collectors.toList());
        float lambdaMin = Float.MAX_VALUE;
        Model modelMin = null;
        for (Model m : models) {
            float lambda = m.getIntersection(P,v);
            if (lambda > 0 && lambda < lambdaMin){
                lambdaMin = lambda;
                modelMin = m;
            }
        }
        if (modelMin == null)
            return Color.black;
        Vec3f I = new Vec3f();
        Vec3f lambdaMinV = new Vec3f();
        lambdaMinV.setScale(lambdaMin,v);
        I.setAdd(P, lambdaMinV);
        Vec3f normal = modelMin.getNormal(I);
        boolean inside = false;
        if (v.dotProduct(normal) > 0) {
            normal = normal.inverse();
            inside = true;
        }

        if (depth < JavaTga.MAX_RAY_DEPTH) {
            if (modelMin.getReflection() > 0.f) {
                Vec3f r = new Vec3f(v);
                r.sub(normal.subScale(2*normal.dotProduct(v),normal));
                r.normalize();
                float bias = 1e-4f;
                Vec3f biasedI = new Vec3f(I);
                biasedI.addScale(bias,normal);
                Color reflectColor = findColor(I,r,depth+1).scale(modelMin.getReflection());
                color = color.add(reflectColor);
            }
            if (modelMin.getTransparency() > 0.f) {
                float refractIndex;
                if (inside)
                    refractIndex = modelMin.getRefractIndex();
                else refractIndex = 1 / modelMin.getRefractIndex();
                float c1 = normal.dotProduct(v);
                float c2 = (float) Math.sqrt(1 - refractIndex*refractIndex*(1-c1*c1));
                Vec3f t = new Vec3f(v);
                t.scale(refractIndex);
                t.addScale(refractIndex*c1-c2,normal);
                t.normalize();
                float bias = 1e-4f;
                Vec3f biasedI = new Vec3f(I);
                biasedI.subScale(bias,normal);
                Color transColor = findColor(I,t,depth+1).scale(modelMin.getTransparency());
                color = color.add(transColor);
            }
        }

        for (Light l : lights) {
            color = color.add(l.getAmbient().mul(modelMin.getAmbiantMaterial()));
            boolean seen = true;
            Vec3f IS = new Vec3f();
            for (Model m : models) {
                IS.setSub(l.getPosition(), I);
                float bias = 1e-4f;
                I.addScale(bias,normal);
                lambdaMin = m.getIntersection(I, IS);
                if (lambdaMin > 0 && lambdaMin < 1) {
                    seen = false;
                    break;
                }
            }
            if (seen) {
                Color diffuse;
                IS.normalize();
                float weight = Math.max(normal.dotProduct(IS),0.f);
                diffuse = modelMin.getDiffuseMaterial().mul(l.getDiffuse().scale(weight));
                Color specular;
                Vec3f halfdir = new Vec3f();
                halfdir.setAdd(IS,v.inverse());
                halfdir.normalize();
                float spec = (float) Math.pow(Math.max(halfdir.dotProduct(normal),0.f),modelMin.getShininess());
                specular = modelMin.getSpecularMaterial().mul(l.getSpecular()).scale(spec);
                color = color.add(diffuse).add(specular);
            }
        }

        return color;
    }
}
