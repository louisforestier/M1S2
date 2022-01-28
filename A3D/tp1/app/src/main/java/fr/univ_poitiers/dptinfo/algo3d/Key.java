package fr.univ_poitiers.dptinfo.algo3d;

import java.util.Objects;

public class Key implements Comparable<Key> {

    private float x;
    private float y;
    private float z;


    public Key(float x, float y, float z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    private boolean almostEquals(float a, float b) {
        return Math.abs(a - b) < 10E-6;
    }

    @Override
    public int compareTo(Key key) {
        int result;
        if (!almostEquals(this.x, key.x)) {
            if (this.x < key.x)
                result = -1;
            else result = 1;
        } else if (!almostEquals(this.y, key.y)){
            if (this.y < key.y)
                result = -1;
            else result = 1;
        } else if (!almostEquals(this.z, key.z)){
            if (this.z < key.z)
                result = -1;
            else result = 1;
        } else result = 0;
        return result;

    }
}
