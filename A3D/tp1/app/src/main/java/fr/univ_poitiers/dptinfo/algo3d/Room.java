package fr.univ_poitiers.dptinfo.algo3d;


public class Room extends GameObject {


    public Room(boolean[] doors, float length, float width, float height, float[] floorColor, float[] ceilingColor, float[] wallColor) {
        super();
        GameObject floor = new GameObject(floorColor);
        floor.setMesh(Plane.INSTANCE);
        floor.getTransform().scalex(width/10).scalez(length/10);
        GameObject ceiling = new GameObject(ceilingColor);
        ceiling.setMesh(Plane.INSTANCE);
        ceiling.getTransform().posy(height).rotx(180.f).scalex(width/10).scalez(length/10);
        Wall w1 = new Wall(width, height, wallColor, doors[0]);
        w1.getTransform().posz(-length/2);
        Wall w2 = new Wall(length, height, wallColor, doors[1]);
        w2.getTransform().roty(90.f).posx(-width/2);
        Wall w3 = new Wall(width, height, wallColor, doors[2]);
        w3.getTransform().roty(180.f).posz(length/2);
        Wall w4 = new Wall(length, height, wallColor, doors[3]);
        w4.getTransform().roty(270.f).posx(width/2);
        this.addChildren(floor);
        this.addChildren(ceiling);
        this.addChildren(w1);
        this.addChildren(w2);
        this.addChildren(w3);
        this.addChildren(w4);
    }
}

