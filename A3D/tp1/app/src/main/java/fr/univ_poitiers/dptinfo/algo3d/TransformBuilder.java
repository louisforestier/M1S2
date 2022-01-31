package fr.univ_poitiers.dptinfo.algo3d;

public class TransformBuilder {
    private float _posx =0;
    private float _posy =0;
    private float _posz =0;
    private float _rotx =0;
    private float _roty =0;
    private float _rotz =0;
    private float _scalex =0;
    private float _scaley =0;
    private float _scalez =0;

    public TransformBuilder(){}

    public Transform buildTransform(){
        return new Transform(_posx, _posy, _posz, _rotx, _roty, _rotz, _scalex, _scaley, _scalez);
    }

    public TransformBuilder posx(float posx){
        this._posx = posx;
        return this;
    }
    public TransformBuilder posy(float posy){
        this._posy = posy;
        return this;
    }

    public TransformBuilder posz(float posz){
        this._posz = posz;
        return this;
    }

    public TransformBuilder rotx(float rotx){
        this._rotx = rotx;
        return this;
    }

    public TransformBuilder roty(float roty){
        this._roty = roty;
        return this;
    }

    public TransformBuilder rotz(float rotz){
        this._rotz = rotz;
        return this;
    }

    public TransformBuilder scalex(float scalex){
        this._scalex = scalex;
        return this;
    }

    public TransformBuilder scaley(float scaley){
        this._scaley = scaley;
        return this;
    }

    public TransformBuilder scalez(float scalez){
        this._scalez = scalez;
        return this;
    }


}
