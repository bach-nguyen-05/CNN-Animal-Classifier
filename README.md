# Convolutional Neural Network (CNN) on Animal Dataset
## Overview
AnimalCNN is a convolutional neural network (CNN) implemented in PyTorch. It is designed to classify images from the CIFAR-10 dataset provided by <em>torchvision</em>, 
which consists of 60,000 32x32 color images in 10 distinct classes. 
The network includes convolutional layers, pooling, fully connected layers, and regularization.
## Three Convolutional Layers
The image is passed through the first convolutional layer which extracts basic features such as edges and then the layers gradually increase the number of output channels
to capture more complex features. <br>
<strong>Similarities: </strong>
+ Kernel size applied to 3 layers are all 3x3
+ Utilized ReLU as an Activation Function and batch normalization

<strong>Differences: </strong> <br>
  
  <strong>Layer 1</strong>
+ Input: BGR images (3 channels)
+ Output 32 channels  <br>

<strong>Layer 2</strong>
+ Input: 32 channels
+ Output 64 channels  <br>

<strong>Layer 2</strong>
+ Input: 64 channels
+ Output 128 channels  <br>

## Max Pooling
Max Pooling is similar to compressing a file or image or quantizing a large language model for faster computation. <br>
+ Max pooling reduces the spatial dimensions of the feature maps by a factor of 2, which helps in reducing computation.
+ Here, the output is also compressed into a 1D vector to be fed into the fully connected layers.
## Fully Connected Layers
<strong>Flattening:</strong>

After three pooling layers, the spatial size reduces to 4x4. The 128 feature maps are flattened into a vector of size 128 * 4 * 4.

<strong>Fully Connected Layer 1:</strong>

+ Input: Flattened vector (2048 elements)
+ Output: 256 neurons with ReLU activation
+ Regularization: Dropout (50%) to reduce overfitting

<strong>Fully Connected Layer 2:</strong>

+ Output: 10 neurons corresponding to the 10 CIFAR-10 classes
