package com.sunkenhero.core.layer;

public class Convolutional3D implements Layer {

    private static final long serialVersionUID = 2564130890068637316L;

    private int filterSizeX;
    private int filterSizeY;
    private int inputWidth;
    private int inputHeight;
    private int inputChannel;
    private int strideX;
    private int strideY;
    private int outputWidth;
    private int outputHeight;
    private float[][][] filter;
    private float[] biases;

    public Convolutional3D(
            int filterSizeX, int filterSizeY,
            int inputWidth, int inputHeight,
            int inputChannel) {
        this(filterSizeX, filterSizeY, inputWidth, inputHeight, inputChannel, 1, 1);
    }

    public Convolutional3D(
            int filterSizeX, int filterSizeY,
            int inputWidth, int inputHeight,
            int inputChannel,
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
        this.inputChannel = inputChannel;
        this.strideX = strideX;
        this.strideY = strideY;
        this.outputWidth = (inputWidth - filterSizeX) / strideX + 1;
        this.outputHeight = (inputHeight - filterSizeY) / strideY + 1;
        if (outputWidth <= 0 || outputHeight <= 0) {
            throw new IllegalArgumentException("Input size too small");
        }
        filter = new float[inputChannel][filterSizeY][filterSizeX];
        biases = new float[inputChannel];
        float stddev = (float) Math.sqrt(2);
        for (int ic = 0; ic < inputChannel; ic++) {
            for (int i = 0; i < filterSizeY; i++) {
                for (int j = 0; j < filterSizeX; j++) {
                    filter[ic][i][j] = (float) Math.random() * stddev;
                }
            }
        }
    }

    @Override
    public float[] forward(float[] input) {
        if (input.length != inputWidth * inputHeight * inputChannel) {
            throw new IllegalArgumentException("Input size does not match layer size");
        }

        int index = 0;
        float[] output = new float[outputHeight * outputWidth * inputChannel];

        for (int ic = 0; ic < inputChannel; ic++) {
            float[][] input2D = new float[inputHeight][inputWidth];
            float[][] output2D = new float[outputHeight][outputWidth];

            int inputOffset = ic * inputWidth * inputHeight;
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    input2D[i][j] = input[inputOffset++];
                }
            }

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    float sum = 0;
                    for (int fi = 0; fi < filterSizeY; fi++) {
                        for (int fj = 0; fj < filterSizeX; fj++) {
                            sum += input2D[i * strideY + fi][j * strideX + fj] * filter[ic][fi][fj];
                        }
                    }
                    output2D[i][j] = sum + biases[ic];
                }
            }

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    output[index++] = output2D[i][j];
                }
            }
        }

        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        int dOutputIndex = 0;
        float[] dInput = new float[inputWidth * inputHeight * inputChannel];

        for (int ic = 0; ic < inputChannel; ic++) {
            float[][] dOutput2D = new float[outputHeight][outputWidth];
            float[][] input2D = new float[inputHeight][inputWidth];
            float[][] dInput2D = new float[inputHeight][inputWidth];
            float[][] dFilter = new float[filterSizeY][filterSizeX];
            float dBias = 0;

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    dOutput2D[i][j] = dOutput[dOutputIndex++];
                }
            }

            int inputOffset = ic * inputWidth * inputHeight;
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    input2D[i][j] = input[inputOffset++];
                }
            }

            for (int i = 0; i < outputHeight; i++) {
                for (int j = 0; j < outputWidth; j++) {
                    float dO = dOutput2D[i][j];
                    dBias += dO;

                    for (int fi = 0; fi < filterSizeY; fi++) {
                        for (int fj = 0; fj < filterSizeX; fj++) {
                            int inputRow = i * strideY + fi;
                            int inputCol = j * strideX + fj;
                            dFilter[fi][fj] += input2D[inputRow][inputCol] * dO;
                            dInput2D[inputRow][inputCol] += filter[ic][fi][fj] * dO;
                        }
                    }
                }
            }

            int dInputOffset = ic * inputWidth * inputHeight;
            for (int i = 0; i < inputHeight; i++) {
                for (int j = 0; j < inputWidth; j++) {
                    dInput[dInputOffset++] = dInput2D[i][j];
                }
            }

            for (int fi = 0; fi < filterSizeY; fi++) {
                for (int fj = 0; fj < filterSizeX; fj++) {
                    filter[ic][fi][fj] -= learningRate * dFilter[fi][fj];
                }
            }

            biases[ic] -= learningRate * dBias;
        }

        return dInput;
    }
}