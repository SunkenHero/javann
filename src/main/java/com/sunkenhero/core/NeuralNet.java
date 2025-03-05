package com.sunkenhero.core;

import java.io.*;
import java.util.*;

import com.sunkenhero.core.layer.Layer;
import com.sunkenhero.core.loss.Loss;

public class NeuralNet implements Serializable {

    private static final long serialVersionUID = 506151823693114784L;

    private final List<Layer> layers;
    private final List<float[]> layerInputs;
    private final List<float[]> layerOutputs;

    public NeuralNet() {
        layers = new ArrayList<>();
        layerInputs = new ArrayList<>();
        layerOutputs = new ArrayList<>();
    }

    public void addLayers(Layer... layer) {
        Collections.addAll(layers, layer);
    }

    public float[] predict(float[] input) {
        layerInputs.clear();
        layerOutputs.clear();
        float[] currentInput = input;
        for (Layer layer : layers) {
            layerInputs.add(currentInput);
            currentInput = layer.forward(currentInput);
            layerOutputs.add(currentInput);
        }
        return currentInput;
    }

    public void backward(float[] dOutput, float learningRate) {
        float[] currentDOutput = dOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            Layer layer = layers.get(i);
            float[] input = layerInputs.get(i);
            float[] output = layerOutputs.get(i);
            currentDOutput = layer.backward(currentDOutput, input, output, learningRate);
        }
    }

    public void train(float[][] inputs, float[][] targets, Loss lossFunction, int epochs, float learningRate) {
        train(inputs, targets, lossFunction, epochs, learningRate, inputs.length);
    }

    public void train(float[][] inputs, float[][] targets, Loss lossFunction, int epochs, float learningRate, int max) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < Math.min(inputs.length, max); i++) {
                backward(lossFunction.gradient(predict(inputs[i]), targets[i]), learningRate);
            }
        }
    }

    public float test(float[][] inputs, float[][] targets, Loss lossFunction) {
        return test(inputs, targets, lossFunction, inputs.length);
    }

    public float test(float[][] inputs, float[][] targets, Loss lossFunction, int max) {
        float loss = 0;
        for (int i = 0; i < Math.min(inputs.length, max); i++) {
            loss += lossFunction.loss(predict(inputs[i]), targets[i]);
        }
        return loss / Math.min(inputs.length, max);
    }

    public void save(String filename) {
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(filename);
            save(fileOutputStream);
            fileOutputStream.close();
        } catch (Exception e) {
            throw new RuntimeException("Failed to save " + filename);
        }

    }

    public void save(OutputStream stream) throws IOException {
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(stream);
        objectOutputStream.writeObject(this);
        objectOutputStream.flush();
        objectOutputStream.close();
    }

    public static NeuralNet load(String filename) {
        try {
            FileInputStream fileInputStream = new FileInputStream(filename);
            NeuralNet net = load(fileInputStream);
            fileInputStream.close();
            return net;
        } catch (Exception e) {
            throw new RuntimeException("Failed to load " + filename);
        }
    }

    public static NeuralNet load(InputStream stream) throws IOException, ClassNotFoundException {
        ObjectInputStream objectInputStream = new ObjectInputStream(stream);
        NeuralNet net = (NeuralNet) objectInputStream.readObject();
        objectInputStream.close();
        return net;
    }

    @Override
    public String toString() {
        String str = getClass().getSimpleName() + "(";
        for (Layer layer : layers) {
            str += "\n\t" + layer.toString().replace("\n", "\n\t");
        }
        return str + "\n)";
    }

}
