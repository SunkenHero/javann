package com.sunkenhero.core.layer;

public class Convolutional2D implements Layer {

    private static final long serialVersionUID = -6914020296840420510L;

    private int filterSizeX;
    private int filterSizeY;
    private int inputWidth;
    private int inputHeight;
    private int strideX;
    private int outputWidth;
    private int outputHeight;
    private int strideOffset;
    private float[][] filter;
    private float bias;

    public Convolutional2D(
            int filterSizeX, int filterSizeY,
            int inputWidth, int inputHeight) {
        this(filterSizeX, filterSizeY, inputWidth, inputHeight, 1, 1);
    }

    public Convolutional2D(
            int filterSizeX, int filterSizeY,
            int inputWidth, int inputHeight,
            int strideX, int strideY) {
        if (filterSizeX <= 1 || filterSizeY <= 1) {
            throw new IllegalArgumentException("Invalid filter size");
        }
        if (inputWidth <= 1 || inputHeight <= 1) {
            throw new IllegalArgumentException("Invalid size");
        }
        if (strideX < 1 || strideY < 1) {
            throw new IllegalArgumentException("Invalid stride");
        }
        this.filterSizeX = filterSizeX;
        this.filterSizeY = filterSizeY;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.strideX = strideX;
        this.outputWidth = (inputWidth - filterSizeX) / strideX + 1;
        this.outputHeight = (inputHeight - filterSizeY) / strideY + 1;
        this.strideOffset = inputWidth - (outputWidth * strideX - strideX + filterSizeX) + inputWidth * (strideY - 1)
                + 1;
        if (outputWidth <= 0 || outputHeight <= 0) {
            throw new IllegalArgumentException("Input size to small");
        }
        filter = new float[filterSizeY][filterSizeX];
        bias = 0;
        float stddev = (float) Math.sqrt(2);
        for (int i = 0; i < filterSizeY; i++) {
            for (int j = 0; j < filterSizeX; j++) {
                filter[i][j] = (float) Math.random() * stddev;
                filter[i][j] = (int) (Math.random() * 2);
            }
        }
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != inputWidth * inputHeight) {
            throw new IllegalArgumentException("Input size does not match layer size");
        }

        int oIndex = 0;
        int iIndex = 0;
        float[] output = new float[outputHeight * outputWidth];
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                float sum = 0;
                for (int fi = 0; fi < filterSizeY; fi++) {
                    for (int fj = 0; fj < filterSizeX; fj++) {
                        sum += input[iIndex + fi * inputWidth + fj] * filter[fi][fj];
                    }
                }
                output[oIndex++] = sum + bias;
                iIndex += strideX;
            }
            iIndex += strideOffset;
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[inputHeight * inputWidth];
        float[][] dFilter = new float[filterSizeY][filterSizeX];
        float dBias = 0;

        int oIndex = 0;
        int iIndex = 0;
        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                float dO = dOutput[oIndex++];
                dBias += dO;
                for (int fi = 0; fi < filterSizeY; fi++) {
                    for (int fj = 0; fj < filterSizeX; fj++) {
                        int index = iIndex + fi * inputWidth + fj;
                        dFilter[fi][fj] += input[index] * dO;
                        dInput[index] += filter[fi][fj] * dO;
                    }
                }
                iIndex += strideX;
            }
            iIndex += strideOffset;
        }

        for (int fi = 0; fi < filterSizeY; fi++) {
            for (int fj = 0; fj < filterSizeX; fj++) {
                filter[fi][fj] -= learningRate * dFilter[fi][fj];
            }
        }

        bias -= learningRate * dBias;

        return dInput;
    }

}
