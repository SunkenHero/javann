package com.sunkenhero.core.loss;

public class MSELoss implements Loss {

    @Override
    public float loss(float[] prediction, float[] target) {
        float sum = 0;
        for (int i = 0; i < prediction.length; i++) {
            float diff = prediction[i] - target[i];
            sum += diff * diff;
        }
        return sum / prediction.length;
    }

    @Override
    public float[] gradient(float[] prediction, float[] target) {
        float[] grad = new float[prediction.length];
        for (int i = 0; i < prediction.length; i++) {
            grad[i] = 2 * (prediction[i] - target[i]) / prediction.length;
        }
        return grad;
    }

}
