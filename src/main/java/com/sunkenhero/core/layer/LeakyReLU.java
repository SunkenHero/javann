package com.sunkenhero.core.layer;

public class LeakyReLU extends ActivationFunction {

    private static final long serialVersionUID = 1376140717779515512L;

    public final float alpha;

    public LeakyReLU(float alpha) {
        this.alpha = alpha;
    }

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] >= 0 ? input[i] : alpha * input[i];
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            dInput[i] = input[i] > 0 ? dOutput[i] : alpha * dOutput[i];
        }
        return dInput;
    }

}
