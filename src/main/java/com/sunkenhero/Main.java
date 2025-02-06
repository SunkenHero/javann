package com.sunkenhero;

import java.util.Arrays;

import com.sunkenhero.core.NeuralNet;
import com.sunkenhero.core.layer.*;

public class Main {

    public Main() {
        NeuralNet net = new NeuralNet();
        net.addLayers(
                new DenseLayer(2, 3),
                new Sigmoid(),
                new DenseLayer(3, 1),
                new Sigmoid());

        float[] output = net.predict(new float[] { 0.5f, 0.3f });
        System.out.println(Arrays.toString(output));
        System.out.println(net);

        try {
            net.save("models/network.jnn");
            net = NeuralNet.load("models/network.jnn");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new Main();
    }

}
