# NeuralLib


`NeuralLib` is a Python library designed for advanced biosignal processing using neural networks. The primary objective is to establish a modular, efficient, generalizable framework for biosignal processing using DL.
The core concept of `NeuralLib` revolves around creating, training, and managing neural network models and leveraging their components for transfer learning (TL). This allows for the reusability of pre-trained models or parts of them to create new models and adapt them to different tasks or datasets efficiently.

The library supports:

- Training and testing `Architectures` from scratch for specific biosignals processing tasks.
- Adding tested models to hugging face repositories to create `ProductionModels` and share them with the community for public usage.
- Extracting trained components from production models using `TLFactory`.
- Combining, freezing, or further fine-tuning pre-trained components to train`TLModels`.


## Tutorials

Explore the [`tutorials/`](./tutorials) folder for several hands-on examples demonstrating how to use the core functionalities of NeuralLib.

## 📖 Documentation

Comprehensive documentation is available here:  
[NeuralLib Documentation](https://novabiosignals.github.io/NeuralLib-docs/)

## Pre-trained Models

Collection of pre-trained models on Hugging Face:  
[NeuralLib DL Models for Biosignals](https://huggingface.co/collections/marianaagdias/neurallibdeep-learning-models-for-biosignals-processing-67473f72e30e1f0874ec5ebe)