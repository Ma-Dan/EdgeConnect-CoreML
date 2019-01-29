# EdgeConnect-CoreML

An example of [EdgeConnect Model](https://github.com/knazeri/edge-connect) on iOS using CoreML.

![AppUI](./images/edgeconnect.gif)

## About EdgeConnect

[EdgeConnect](https://github.com/knazeri/edge-connect) is a Generative Image Inpainting with Adversarial Edge Learning. [https://arxiv.org/abs/1901.00212](https://arxiv.org/abs/1901.00212)

## Code references
- UI code: [NSTDemo](https://github.com/kirualex/NSTDemo)
- CoreMLHelpers code: [CoreMLHelpers](https://github.com/hollance/CoreMLHelpers)

## Usage

How to convert pytorch model to CoreML model:

1. Run [modified edge-connect code](https://github.com/Ma-Dan/edge-connect) to remove spectral normalization and convert to ONNX model.

```shell
python test.py --checkpoints ./checkpoints/places2 --input ./examples/test/places2 --output ./results
```
Modify [line 158](https://github.com/Ma-Dan/edge-connect/blob/master/src/models.py#L158) to change input resolution.

Modify checkpoint argument to use other pretrained weights. This example uses places2 weight.

2. Run [ONNX to CoreML converter](https://github.com/onnx/onnx-coreml) on ONNX files to get CoreML models.

## Todo

1. Add Canny edge detecion preprocess to get more accurate edge output.
2. Apply mask from touch screen input.
