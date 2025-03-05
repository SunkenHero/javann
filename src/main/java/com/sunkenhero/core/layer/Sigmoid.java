package com.sunkenhero.core.layer;

public class Sigmoid extends ActivationFunction {

    private static final long serialVersionUID = -1413449610535490852L;

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = 1 / (1 + (float) Math.exp(-input[i]));
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            dInput[i] = dOutput[i] * output[i] * (1 - output[i]);
        }
        return dInput;
    }

}
