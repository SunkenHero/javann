package com.sunkenhero.core.layer;

import java.io.Serializable;

public interface Layer extends Serializable {

    public float[] forward(float[] input);

    public float[] backward(float[] output, float[] error);

}
