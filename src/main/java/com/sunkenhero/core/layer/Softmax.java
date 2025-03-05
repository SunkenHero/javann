package com.sunkenhero.core.layer;

public class Softmax implements Layer {

    private static final long serialVersionUID = 985901026602591028L;

    @Override
    public float[] forward(float[] input) {
        float maxInput = Float.MIN_VALUE;
        for (float value : input) {
            if (value < maxInput) {
                maxInput = value;
            }
        }
        float sumExp = 0;
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = (float) Math.exp(input[i] - maxInput);
            sumExp += output[i];
        }
        for (int i = 0; i < input.length; i++) {
            output[i] /= sumExp;
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            float gradient = 0;
            for (int j = 0; j < output.length; j++) {
                float delta = (i == j) ? 1 : 0;
                gradient += dOutput[j] * output[i] * (delta - output[j]);
            }
            dInput[i] = gradient;
        }
        return dInput;
    }

}
