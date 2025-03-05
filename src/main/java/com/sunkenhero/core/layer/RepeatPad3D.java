package com.sunkenhero.core.layer;

public class RepeatPad3D implements Layer {

    private static final long serialVersionUID = 2795848288686798418L;

    private int inputWidth;
    private int inputHeight;
    private int inputChannel;
    private int paddingLeft;
    private int paddingTop;
    private int paddedWidth;
    private int paddedHeight;

    public RepeatPad3D(int inputWidth, int inputHeight, int inputChannel, int padding) {
        this(inputWidth, inputHeight, inputChannel, padding, padding);
    }

    public RepeatPad3D(int inputWidth, int inputHeight, int inputChannel, int paddingX, int paddingY) {
        this(inputWidth, inputHeight, inputChannel, paddingX, paddingX, paddingY, paddingY);
    }

    public RepeatPad3D(int inputWidth, int inputHeight, int inputChannel, int paddingLeft, int paddingRight,
            int paddingTop,
            int paddingBottom) {

        if (inputWidth <= 0 || inputHeight <= 0) {
            throw new IllegalArgumentException("Invalid size");
        }
        if (paddingLeft < 0 || paddingRight < 0 || paddingTop < 0 || paddingBottom < 0) {
            throw new IllegalArgumentException("Invalid padding");
        }
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputChannel = inputChannel;
        this.paddingLeft = paddingLeft;
        this.paddingTop = paddingTop;
        paddedWidth = inputWidth + paddingLeft + paddingRight;
        paddedHeight = inputHeight + paddingTop + paddingBottom;
    }

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[paddedHeight * paddedWidth * inputChannel];
        int index = 0;
        for (int ic = 0; ic < inputChannel; ic++) {
            for (int i = 0; i < paddedHeight; i++) {
                for (int j = 0; j < paddedWidth; j++) {
                    int originalI = i - paddingTop;
                    int originalJ = j - paddingLeft;
                    if (originalI >= 0 && originalI < inputHeight && originalJ >= 0 && originalJ < inputWidth) {
                        output[index] = input[originalI * inputWidth + originalJ + ic * inputWidth * inputHeight];
                    } else {
                        int repeatI = (originalI + inputHeight) % inputHeight;
                        int repeatJ = (originalJ + inputWidth) % inputWidth;
                        output[index] = input[repeatI * inputWidth + repeatJ + ic * inputWidth * inputHeight];
                    }
                    index++;
                }
            }
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[inputHeight * inputWidth * inputChannel];
        int index = 0;
        for (int ic = 0; ic < inputChannel; ic++) {
            for (int i = 0; i < paddedHeight; i++) {
                for (int j = 0; j < paddedWidth; j++) {
                    int originalI = i - paddingTop;
                    int originalJ = j - paddingLeft;
                    if (originalI >= 0 && originalI < inputHeight && originalJ >= 0 && originalJ < inputWidth) {
                        dInput[originalI * inputWidth + originalJ + ic * inputWidth * inputHeight] = dOutput[index];
                    }
                    index++;
                }
            }
        }
        return dInput;
    }

}
