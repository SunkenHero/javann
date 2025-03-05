package com.sunkenhero.core.layer;

import java.util.Arrays;

public class PReLU extends ActivationFunction {

    private static final long serialVersionUID = 4569111330499998982L;

    private final int numInputs;
    private final float[] alphas;

    public PReLU(int numInputs) {
        this(numInputs, 0.25f);
    }

    public PReLU(int numInputs, float init) {
        this.numInputs = numInputs;
        alphas = new float[numInputs];
        Arrays.fill(alphas, init);
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != numInputs) {
            throw new IllegalArgumentException("Input size must match layers input size");
        }
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (input[i] > 0) ? input[i] : alphas[i] * input[i];
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            if (input[i] > 0) {
                dInput[i] = dOutput[i];
            } else {
                dInput[i] = alphas[i] * dOutput[i];
                alphas[i] -= learningRate * dOutput[i] * input[i];
            }
        }
        return dInput;
    }

    @Override
    public String toString() {
        return Arrays.toString(alphas);
    }

}
