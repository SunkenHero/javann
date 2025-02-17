package com.sunkenhero;

import java.util.Arrays;

import com.sunkenhero.core.NeuralNet;
import com.sunkenhero.core.layer.*;

public class Main {

    public Main() {
        NeuralNet net = new NeuralNet();

        boolean load = false;
        if (load) {
            try {
                net = NeuralNet.load("models/network.jnn");
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            net.addLayers(
                    new DenseLayer(3, 4),
                    new LeakyReLU(0.1f),
                    new DenseLayer(4, 2),
                    new LeakyReLU(0.1f));
        }

        float[] input = { 0.5f, 0.8f, -0.2f };
        float[] target = { 1.0f, 0.0f };

        for (int epoch = 0; epoch < 1000; epoch++) {
            float[] output = net.predict(input);

            float[] dOutput = new float[output.length];
            for (int i = 0; i < output.length; i++) {
                dOutput[i] = 2 * (output[i] - target[i]);
            }

            net.backward(dOutput, 0.01f);

            if (epoch % 100 == 0) {
                System.out.println("Epoch " + epoch + ", Output: " + Arrays.toString(output));
            }
        }

        try {
            net.save("models/network.jnn");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        new Main();
    }

}
