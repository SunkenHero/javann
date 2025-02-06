package com.sunkenhero.core.layer;

public abstract class ActivationFunction implements Layer {

    private static final long serialVersionUID = 6856743699842568763L;

    @Override
    public String toString() {
        return getClass().getSimpleName() + "()";
    }

}
