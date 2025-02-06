package com.sunkenhero.core.layer;

public class DenseLayer implements Layer {

    private static final long serialVersionUID = -5733950919604580232L;

    private final Neuron[] neurons;

    private final int numInputs;
    private final int numNeurons;

    public DenseLayer(int numInputs, int numNeurons) {
        if (numNeurons <= 0) {
            throw new IllegalArgumentException("Number of neurons must be greater than zero.");
        }
        this.numInputs = numInputs;
        this.numNeurons = numNeurons;
        this.neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            neurons[i] = new Neuron(numInputs);
        }
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != numInputs) {
            throw new IllegalArgumentException("Input size must match neuron input size.");
        }
        float[] outputs = new float[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            outputs[i] = neurons[i].forward(input);
        }
        return outputs;
    }

    @Override
    public float[] backward(float[] output, float[] error) {
        throw new UnsupportedOperationException("Unimplemented method 'backward'");
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    @Override
    public String toString() {
        String str = getClass().getSimpleName() + "(";
        if (numNeurons <= 5) {
            for (Neuron neuron : neurons) {
                str += "\n\t" + neuron;
            }
        } else {
            for (int i = 0; i < 3; i++) {
                str += "\n\t" + neurons[i];
            }
            str += "\n\t...\n\t" + neurons[numNeurons - 1];
        }
        return str + "\n)";
    }

}
