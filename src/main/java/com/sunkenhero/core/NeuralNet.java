package com.sunkenhero.core;

import java.io.*;
import java.util.*;

import com.sunkenhero.core.layer.Layer;

public class NeuralNet implements Serializable {

    private static final long serialVersionUID = 506151823693114784L;

    private final ArrayList<Layer> layers;

    public NeuralNet() {
        layers = new ArrayList<>();
    }

    public void addLayers(Layer... layer) {
        Collections.addAll(layers, layer);
    }

    public float[] predict(float[] input) {
        float[] output = input;
        for (Layer layer : layers) {
            output = layer.forward(output);
        }
        return output;
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
