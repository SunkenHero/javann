package com.sunkenhero.core.layer;

public class Tanh extends ActivationFunction {

    private static final long serialVersionUID = -6626120325742486619L;

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.tanh(input[i]);
        }
        return output;
    }

    @Override
    public float[] backward(float[] output, float[] error) {
        float[] gradients = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            gradients[i] = error[i] * (1 - output[i] * output[i]);
        }
        return gradients;
    }

}
