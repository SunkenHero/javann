package com.sunkenhero.core.loss;

public interface Loss {

    public float loss(float[] prediction, float[] target);

    public float[] gradient(float[] prediction, float[] target);

}
