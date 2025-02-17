package com.sunkenhero.core;

import java.io.*;
import java.util.*;

import com.sunkenhero.core.layer.Layer;

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
            float[] output = layer.forward(currentInput);
            layerOutputs.add(output);
            currentInput = output;
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

    public void save(String filename) throws IOException {
        FileOutputStream fileOutputStream = new FileOutputStream(filename);
        save(fileOutputStream);
        fileOutputStream.close();
    }

    public void save(OutputStream stream) throws IOException {
        ObjectOutputStream objectOutputStream = new ObjectOutputStream(stream);
        objectOutputStream.writeObject(this);
        objectOutputStream.flush();
        objectOutputStream.close();
    }

    public static NeuralNet load(String filename) throws IOException, ClassNotFoundException {
        FileInputStream fileInputStream = new FileInputStream(filename);
        NeuralNet net = load(fileInputStream);
        fileInputStream.close();
        return net;
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
