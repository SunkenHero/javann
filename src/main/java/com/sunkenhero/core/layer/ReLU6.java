package com.sunkenhero.core.layer;

public class ReLU6 extends ActivationFunction {

    private static final long serialVersionUID = 4569111330499998982L;

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.clamp(input[i], 0, 6);
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            dInput[i] = (input[i] < 0 || input[i] > 6) ? 0 : dOutput[i];
        }
        return dInput;
    }

}
