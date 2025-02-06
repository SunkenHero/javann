package com.sunkenhero.core.layer;

public class Sigmoid extends ActivationFunction {

    private static final long serialVersionUID = -1413449610535490852L;

    @Override
    public float[] forward(float[] input) {
        for (int i = 0; i < input.length; i++) {
            input[i] = 1 / (1 + (float) Math.exp(-input[i]));
        }
        return input;
    }

    @Override
    public float[] backward(float[] output, float[] error) {
        float[] gradient = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[i] = error[i] * output[i] * (1 - output[i]);
        }
        return gradient;
    }

}
