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
    public float[] backward(float[] output, float[] error) {
        float[] gradients = new float[output.length];
        for (int i = 0; i < output.length; i++) {
            gradients[i] = output[i] >= 0 ? error[i] : error[i] * (output[i] + alpha);
        }
        return gradients;
    }

}
