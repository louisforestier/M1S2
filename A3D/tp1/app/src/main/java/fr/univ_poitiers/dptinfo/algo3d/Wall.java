package fr.univ_poitiers.dptinfo.algo3d;

public class Wall extends GameObject {


    public Wall(float width, float height, float[] color, boolean hasDoor) {
        super(color);
        getTransform().posy(height/2);
        if (!hasDoor) {
            GameObject plane = new GameObject(color);
            plane.setMesh(Plane.INSTANCE);
            plane.getTransform().scalex(width/10).scalez(height/10).rotx(90.f);
            this.addChildren(plane);
        } else {
            GameObject left = new GameObject(color);
            left.setMesh(Plane.INSTANCE);
            left.getTransform().scalex(((width-1.f)/2)/10).scalez(height/10).posx(-0.5f-(width-1.f)/4).rotx(90.f);
            this.addChildren(left);

            GameObject middle = new GameObject(color);
            middle.setMesh(Plane.INSTANCE);
            middle.getTransform().scalex(1.f/10).scalez((height-2.f)/10).posy(2.f-(height/2)+(height-2.f)/2).rotx(90.f);
            this.addChildren(middle);

            GameObject right = new GameObject(color);
            right.setMesh(Plane.INSTANCE);
            right.getTransform().scalex(((width-1.f)/2)/10).scalez(height/10).posx(0.5f+(width-1.f)/4).rotx(90.f);
            this.addChildren(right);
        }
    }
}
