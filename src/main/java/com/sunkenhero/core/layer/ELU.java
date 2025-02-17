package com.sunkenhero.core.layer;

public class ELU extends ActivationFunction {

    private static final long serialVersionUID = -323465809987355596L;

    public final float alpha;

    public ELU(float alpha) {
        this.alpha = alpha;
    }

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            output[i] = input[i] >= 0 ? input[i] : alpha * ((float) Math.exp(input[i]) - 1);
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            float elu = (input[i] > 0) ? 1 : alpha * (float) Math.exp(input[i]);
            dInput[i] = dOutput[i] * elu;
        }
        return dInput;
    }

}
