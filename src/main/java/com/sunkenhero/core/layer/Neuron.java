package com.sunkenhero.core.layer;

import java.io.Serializable;

public class Neuron implements Serializable {

    private static final long serialVersionUID = 7447794927150606679L;

    private float bias;
    private float[] weights;

    private final int numInputs;

    public Neuron(int numInputs) {
        if (numInputs <= 0) {
            throw new IllegalArgumentException("Number of inputs must be greater than zero");
        }
        this.numInputs = numInputs;
        this.bias = 0;
        this.weights = new float[numInputs];
        float k = (float) Math.sqrt(1.0f / numInputs);
        for (int i = 0; i < numInputs; i++) {
            weights[i] = ((float) Math.random() * 2 - 1) * k;
        }
    }

    public float forward(float[] input) {
        if (input.length != numInputs) {
            throw new IllegalArgumentException("Number of inputs must match neuron's input count");
        }
        float sum = bias;
        for (int i = 0; i < numInputs; i++) {
            sum += weights[i] * input[i];
        }
        return sum;
    }

    public float getBias() {
        return bias;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public float[] getWeights() {
        return weights;
    }

    public void setWeights(float[] weights) {
        if (weights.length != numInputs) {
            throw new IllegalArgumentException("Number of weights must match number of inputs");
        }
        this.weights = weights;
    }

    public int getNumInputs() {
        return numInputs;
    }

    @Override
    public String toString() {
        String str = getClass().getSimpleName() + "( weights=[ ";
        if (numInputs <= 5) {
            for (int i = 0; i < numInputs; i++) {
                str += weights[i];
                if (i != numInputs - 1) {
                    str += ", ";
                }
            }
        } else {
            for (int i = 0; i < 3; i++) {
                str += weights[i] + ", ";
            }
            str += "..., " + weights[numInputs - 1];
        }
        return str + " ], bias=" + bias + " )";
    }

}
