package com.sunkenhero.core.layer;

public class MirrorPad2D implements Layer {

    private static final long serialVersionUID = 3187451868166684488L;

    private int inputWidth;
    private int inputHeight;
    private int paddingLeft;
    private int paddingTop;
    private int paddedWidth;
    private int paddedHeight;

    public MirrorPad2D(int inputWidth, int inputHeight, int padding) {
        this(inputWidth, inputHeight, padding, padding);
    }

    public MirrorPad2D(int inputWidth, int inputHeight, int paddingX, int paddingY) {
        this(inputWidth, inputHeight, paddingX, paddingX, paddingY, paddingY);
    }

    public MirrorPad2D(int inputWidth, int inputHeight, int paddingLeft, int paddingRight, int paddingTop,
            int paddingBottom) {
        if (inputWidth <= 0 || inputHeight <= 0) {
            throw new IllegalArgumentException("Invalid size");
        }
        if (paddingLeft < 0 || paddingRight < 0 || paddingTop < 0 || paddingBottom < 0) {
            throw new IllegalArgumentException("Invalid padding");
        }
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.paddingLeft = paddingLeft;
        this.paddingTop = paddingTop;
        paddedWidth = inputWidth + paddingLeft + paddingRight;
        paddedHeight = inputHeight + paddingTop + paddingBottom;
    }

    @Override
    public float[] forward(float[] input) {
        float[] output = new float[paddedHeight * paddedWidth];
        int index = 0;
        for (int i = 0; i < paddedHeight; i++) {
            for (int j = 0; j < paddedWidth; j++) {
                int originalI = i - paddingTop;
                int originalJ = j - paddingLeft;
                if (originalI >= 0 && originalI < inputHeight && originalJ >= 0 && originalJ < inputWidth) {
                    output[index] = input[originalI * inputWidth + originalJ];
                } else {
                    output[index] = input[Math.abs(originalI % inputHeight) * inputWidth
                            + Math.abs(originalJ % inputWidth)];
                }
                index++;
            }
        }
        return output;
    }

    @Override
    public float[] backward(float[] dOutput, float[] input, float[] output, float learningRate) {
        float[] dInput = new float[inputHeight * inputWidth];
        int index = 0;
        for (int i = 0; i < paddedHeight; i++) {
            for (int j = 0; j < paddedWidth; j++) {
                int originalI = i - paddingTop;
                int originalJ = j - paddingLeft;
                if (originalI >= 0 && originalI < inputHeight && originalJ >= 0 && originalJ < inputWidth) {
                    dInput[originalI * inputWidth + originalJ] = dOutput[index];
                }
                index++;
            }
        }
        return dInput;
    }

}
