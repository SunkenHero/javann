package com.sunkenhero.core.layer;

public class ReLU extends ActivationFunction {

    private static final long serialVersionUID = 4569111330499998982L;

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.max(0, input[i]);
        }
        return output;
    }

    @Override
    public float[] backward(float[] output, float[] error) {
        float[] gradient = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            gradient[i] = output[i] > 0 ? error[i] : 0;
        }
        return gradient;
    }

}
