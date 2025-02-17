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
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            dInput[i] = dOutput[i] * (1 - output[i] * output[i]);
        }
        return dInput;
    }

}
