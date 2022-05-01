package fr.algo3d.model;

import fr.algo3d.model.models.*;
import fr.algo3d.model.lights.*;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Class to represent a scene to be rendered.
 */
public class Scene {

    /**
     * Maximum  number of recursion for the higher order rays.
     */
    public static final int MAX_RAY_DEPTH = 13;

    /**
     * Ambient color of the scene.
     */
    private Color ambient;

    /**
     * List of models in the scene.
     */
    private List<Model> models = new ArrayList<>();

    /**
     * List of lights in the scene.
     */
    private List<Light> lights = new ArrayList<>();

    /**
     * Constructor.
     * Creates a scene with one infinite plane below and one above the point of view, and five spheres.
     */
    public Scene() {
        ambient = Color.darkgray;
        Material green = new Material(Color.green,Color.white,32,0,01.f,1);
        Material orange = new Material(Color.orange,Color.white,100,01.f,01.f,04f);
        Material cyan = new Material(Color.cyan,Color.white,32,0.f,01.f,1);
        Material red = new Material(Color.red,Color.white,32,0.f,01.f,1);
        Material yellow = new Material(Color.yellow,Color.white,32,0.f,01,1);
        models.add(new Plane(green,new Vec3f(0,1,0),-2));
        models.add(new Plane(cyan,new Vec3f(0,1,0),2));
        models.add(new Sphere(orange,new Vec3f(0,0,-7.5f),2));
        models.add(new Sphere(red,new Vec3f(-4,-1,-7.5f),1));
        models.add(new Sphere(yellow,new Vec3f(4,-1,-7.5f),1));
        models.add(new Sphere(red,new Vec3f(3f,-1,-12.5f),1));
        models.add(new Sphere(yellow,new Vec3f(-3f,-1,-12.5f),1));
        lights.add(new Light(new Vec3f(-1,1,0), Color.lightgray,Color.white));
    }

    /**
     * Returns the color found by the ray defined by P and v.
     * Looks if the ray intersects a model. If no then the color is black.
     * Else, it calculates the diffuse and specular color for each light if it is not obscured in a shadow.
     * Then it calculates the reflected and refracted colors.
     * @param P
     * @param v
     * @param depth
     * @return
     */
    public Color findColor(Vec3f P, Vec3f v, int depth){
        Color color = new Color();
        float lambdaMin = Float.MAX_VALUE;
        Model modelMin = null;
        for (Model m : models) {
            float lambda = m.getIntersection(P,v);
            if (lambda > 1E-3 && lambda < lambdaMin){
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
        float reflRatio = modelMin.getReflection() / (1 + modelMin.getReflection()+ modelMin.getTransparency());
        float transRatio = modelMin.getTransparency() / (1 + modelMin.getReflection()+ modelMin.getTransparency());
        float diffuseRatio = 1 / (1 + modelMin.getReflection()+ modelMin.getTransparency());

        for (Light l : lights) {
            color = color.add(ambient.mul(modelMin.getColor()));
            boolean seen = true;
            Vec3f IS = new Vec3f();
            IS.setSub(l.getPosition(), I);
            for (Model m : models) {
                float bias = 1e-4f;
                I.addScale(bias, normal);
                lambdaMin = m.getIntersection(I, IS);
                if (lambdaMin > 0 && lambdaMin < 1) {
                    seen = false;
                    break;
                }
            }
            if (seen) {
                Color diffuse;
                IS.normalize();
                float weight = Math.max(normal.dotProduct(IS), 0.f);
                diffuse = modelMin.getColor().mul(l.getDiffuse().scale(weight));
                diffuse = diffuse.scale(diffuseRatio);
                Color specular;
                Vec3f halfdir = new Vec3f();
                halfdir.setAdd(IS, v.inverse());
                halfdir.normalize();
                float spec = (float) Math.pow(Math.max(halfdir.dotProduct(normal), 0.f), modelMin.getShininess());
                specular = modelMin.getSpecularMaterial().mul(l.getSpecular()).scale(spec);
                specular = specular.scale(diffuseRatio);
                color = color.add(diffuse).add(specular);
            }
        }

        if ((modelMin.getReflection() > 0 || modelMin.getTransparency() > 0) && depth < Scene.MAX_RAY_DEPTH) {
            if (modelMin.getReflection() > 0.f) {
                Vec3f r = new Vec3f(v);
                r.subScale(2*normal.dotProduct(v),normal);
                r.normalize();
                float bias = 1e-4f;
                Vec3f biasedI = new Vec3f(I);
                biasedI.addScale(bias,normal);
                Color reflectColor = findColor(biasedI,r,depth+1).scale(modelMin.getReflection());
                color = color.add(reflectColor.scale(reflRatio));
            }
            if (modelMin.getTransparency() > 0.f) {
                float refractIndex;
                if (inside)
                    refractIndex = modelMin.getRefractIndex();
                else refractIndex = 1 / modelMin.getRefractIndex();
                float c1 = -normal.dotProduct(v);
                float c2 = (float) Math.sqrt(1 - refractIndex*refractIndex*(1-c1*c1));
                Vec3f t = new Vec3f(v);
                t.scale(refractIndex);
                t.addScale(refractIndex*c1-c2,normal);
                t.normalize();
                float bias = 1e-4f;
                Vec3f biasedI = new Vec3f(I);
                biasedI.subScale(bias,normal);
                Color transColor = findColor(biasedI,t,depth+1).scale(modelMin.getTransparency());
                color = color.add(transColor.scale(transRatio));
            }
        }
        return color;
    }

    /**
     * Builds a list of ray in sequential then uses this list in parallel via stream to call findcolor on each element. Uses the transform/map parallel pattern.
     * @param w
     * @param h
     * @param buffer
     * @param image
     */
    public void renderParallelMap(int w, int h, byte[] buffer, byte[] image) {
        List<Vec3f> list = new ArrayList<>();
        for(int row = 0; row < h; row++){ // for each row of the image
            for(int col = 0; col < w; col++) { // for each column of the image
                float x = (col - w / 2.f) / h;
                float y = (row - h / 2.f) / h;
                float z = -1f;
                Vec3f ray = new Vec3f(x,y,z).normalize();
                list.add(ray);
            }
        }
        List<Color> colors = list.parallelStream().map(ray -> findColor(new Vec3f(), ray,0)).collect(Collectors.toList());
        for(int row = 0; row < h; row++) { // for each row of the image
            for (int col = 0; col < w; col++) { // for each column of the image
                int index = 3 * ((row * w) + col); // compute index of color for pixel (x,y) in the buffer
                buffer[index]= (byte) (Math.min(colors.get((row * w) + col).getB(),1.f)*255); // blue : take care, blue is the first component !!!
                buffer[index+1]= (byte) (Math.min(colors.get((row * w) + col).getG(),1.f)*255); // green
                buffer[index+2]= (byte) (Math.min(colors.get((row * w) + col).getR(),1.f)*255); // red (red is the last component !!!)

                image[index]= (byte) (Math.min(colors.get((row * w) + col).getR(),1.f)*255); // Red
                image[index+1]= (byte) (Math.min(colors.get((row * w) + col).getG(),1.f)*255); // green
                image[index+2]= (byte) (Math.min(colors.get((row * w) + col).getB(),1.f)*255); // blue

            }
        }
    }

    /**
     * Builds a map of index and ray in sequential then uses this map in parallel via stream to call findcolor on each element.
     * Better than the precedent algorithm because the index and the ray are calculated in the same loop.
     *
     * @param w
     * @param h
     * @param buffer
     * @param image
     */
    public void renderParallelForEach(int w, int h, byte[] buffer, byte[] image) {
        Map<Integer,Vec3f> map = new HashMap<>();
        for(int row = 0; row < h; row++){ // for each row of the image
            for(int col = 0; col < w; col++) { // for each column of the image
                float x = (col - w / 2.f) / h;
                float y = (row - h / 2.f) / h;
                float z = -1f;
                int index = 3 * ((row * w) + col); // compute index of color for pixel (x,y) in the buffer
                Vec3f ray = new Vec3f(x,y,z).normalize();
                map.put(index,ray);
            }
        }
        map.entrySet().parallelStream().forEach(integerVec3fEntry -> {
            Color c = findColor(new Vec3f(), integerVec3fEntry.getValue(),0);
            buffer[integerVec3fEntry.getKey()]= (byte) (Math.min(c.getB(),1.f)*255); // blue : take care, blue is the first component !!!
            buffer[integerVec3fEntry.getKey()+1]= (byte) (Math.min(c.getG(),1.f)*255); // green
            buffer[integerVec3fEntry.getKey()+2]= (byte) (Math.min(c.getR(),1.f)*255); // red (red is the last component !!!)

            image[integerVec3fEntry.getKey()]= (byte) (Math.min(c.getR(),1.f)*255); // Red
            image[integerVec3fEntry.getKey()+1]= (byte) (Math.min(c.getG(),1.f)*255); // green
            image[integerVec3fEntry.getKey()+2]= (byte) (Math.min(c.getB(),1.f)*255); // blue

        });
    }


    /**
     * Uses IntStream in parallel to mimic nested for loops. Very close to the sequential algorithm in terms of code.
     * Best performance out of all algorithms.
     * @param w
     * @param h
     * @param buffer
     * @param image
     */
    public void renderParallelNestedLoops(int w, int h, byte[] buffer, byte[] image) {
        IntStream.range(0,h).parallel().forEach(row -> {
            IntStream.range(0,w).parallel().forEach(col -> {
                int index = 3*((row*w)+col); // compute index of color for pixel (x,y) in the buffer
                float x = (col - w/2.f)/h;
                float y = (row -h/2.f)/h;
                float z = -1f;
                Color c = findColor(new Vec3f(),(new Vec3f(x,y,z)).normalize(),0);
                // Ensure that the pixel is black

                buffer[index]= (byte) (Math.min(c.getB(),1.f)*255); // blue : take care, blue is the first component !!!
                buffer[index+1]= (byte) (Math.min(c.getG(),1.f)*255); // green
                buffer[index+2]= (byte) (Math.min(c.getR(),1.f)*255); // red (red is the last component !!!)

                image[index]= (byte) (Math.min(c.getR(),1.f)*255); // Red
                image[index+1]= (byte) (Math.min(c.getG(),1.f)*255); // green
                image[index+2]= (byte) (Math.min(c.getB(),1.f)*255); // blue

            });
        });
    }

    /**
     * Classical ray tracing algorithm in sequential.
     * @param w
     * @param h
     * @param buffer
     * @param image
     */
    public void renderSequential(int w, int h, byte[] buffer, byte[] image) {
        for(int row = 0; row < h; row++){ // for each row of the image
            for(int col = 0; col < w; col++){ // for each column of the image

                int index = 3*((row*w)+col); // compute index of color for pixel (x,y) in the buffer
                float x = (col - w/2.f)/h;
                float y = (row -h/2.f)/h;
                float z = -1f;
                Color c = findColor(new Vec3f(),(new Vec3f(x,y,z)).normalize(),0);
                // Ensure that the pixel is black

                buffer[index]= (byte) (Math.min(c.getB(),1.f)*255); // blue : take care, blue is the first component !!!
                buffer[index+1]= (byte) (Math.min(c.getG(),1.f)*255); // green
                buffer[index+2]= (byte) (Math.min(c.getR(),1.f)*255); // red (red is the last component !!!)

                image[index]= (byte) (Math.min(c.getR(),1.f)*255); // Red
                image[index+1]= (byte) (Math.min(c.getG(),1.f)*255); // green
                image[index+2]= (byte) (Math.min(c.getB(),1.f)*255); // blue
            }
        }
    }
}
