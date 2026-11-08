# Project 2: LeNet-5 Reproduction on MNIST

Reproduces the classic LeNet-5 architecture (LeCun et al., 1998) in PyTorch and compares it with a small modern CNN on handwritten digit classification.

## Dataset
MNIST: 60,000 training and 10,000 test images of handwritten digits (0-9). Downloaded automatically with torchvision.

## Models
- **LeNet-5**: 2 convolution layers, tanh activations, average pooling, 3 fully connected layers.
- **Modern CNN**: 2 convolution layers, ReLU activations, max pooling, dropout.

Both were trained with the Adam optimizer (learning rate 1e-3) and cross-entropy loss.

## Results

| Model | Parameters | Test accuracy |
|---|---|---|
| LeNet-5 | 61,706 | 98.73% |
| Modern CNN | 544,522 | 99.12% |

The modern CNN is about 0.4 percentage points more accurate but uses roughly 9x more parameters. LeNet-5 reaches almost the same accuracy with a far smaller model, which shows how strong the original design was.

Confusion matrices and accuracy curves are saved in `outputs_p2/`.

## Limitations
- Trained for a small number of epochs (5), with a single run and no repeated seeds.
- No hyperparameter tuning or data augmentation.

## How to run
```
pip install -r requirements.txt
python p2_lenet_mnist.py --epochs 5
```
