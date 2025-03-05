package com.sunkenhero;

import java.io.*;

import com.sunkenhero.core.NeuralNet;
import com.sunkenhero.core.layer.*;
import com.sunkenhero.core.loss.*;

public class Main {

    public Main() {
        NeuralNet net = new NeuralNet();

        boolean load = true;
        if (load) {
            net = NeuralNet.load("models/network.jnn");
        } else {
            net.addLayers(
                    new ZeroPad2D(28, 28, 1),
                    new Convolutional2D(3, 3, 30, 30),
                    new PReLU(784),

                    new ZeroPad2D(28, 28, 1),
                    new Convolutional2D(3, 3, 30, 30),
                    new PReLU(784),

                    new DenseLayer(784, 512),
                    new PReLU(512),
                    new DenseLayer(512, 256),
                    new PReLU(256),
                    new DenseLayer(256, 128),
                    new PReLU(128),
                    new DenseLayer(128, 64),
                    new PReLU(64),
                    new DenseLayer(64, 32),
                    new PReLU(32),
                    new DenseLayer(32, 10),
                    new ReLU(),

                    new Softmax());
        }

        int numRows = 60000;
        int numClasses = 10;
        int numPixels = 784;

        float[][] inputs = new float[numRows][numPixels];
        float[][] targets = new float[numRows][numClasses];

        try (BufferedReader br = new BufferedReader(new FileReader("datasets/mnist_train.csv"))) {
            String line;
            int rowIndex = 0;

            while ((line = br.readLine()) != null && rowIndex < numRows) {
                String[] values = line.split(",");

                int label = Integer.parseInt(values[0]);
                targets[rowIndex][label] = 1.0f;

                for (int i = 1; i < values.length; i++) {
                    inputs[rowIndex][i - 1] = Float.parseFloat(values[i]) / 255.0f;
                }

                rowIndex++;
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        Loss loss = new MSELoss();
        System.out.println(net.test(inputs, targets, loss, 100));
        net.train(inputs, targets, loss, 1, 0.005f, 60000);
        System.out.println(net.test(inputs, targets, loss, 100));

        net.save("models/network.jnn");
    }

    public static void main(String[] args) {
        new Main();
    }

}
