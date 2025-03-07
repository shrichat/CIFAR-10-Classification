Classification of the CIFAR 10 Dataset, integrating Random Walk Learning with Model Distributed Learning to classify the CIFAR-10 dataset, creating non-IID splits to distribute data heterogeneously across 5 devices, each containing only 2 classes.

Implemented a custom pipeline using PyTorch, ResNet 18 architecture to handle distributed training process, applying a mini
batch SGD of size 2 and executing the algorithm for over 20,000 epochs, with regular checkpointing.

Capable of achieving a validation accuracy of over 85% .
