# Deep Learning (Transfer Learning for Pet Breed Classification)

A comparative study of training from scratch, feature extraction, fine-tuning, data augmentation, and EfficientNet-B0 for image classification with limited labeled data.

**Course:** Deep Learning - Session 05: Advanced CNNs and Transfer Learning  
**Framework:** PyTorch / Torchvision  
**Environment:** Google Colab (NVIDIA T4 GPU)

## Overview

This project investigates different training strategies for classifying pet breeds using a relatively small dataset. The main objective is to understand how pretrained representations affect model performance, training efficiency, and generalization.

ResNet-18 is used for the main experiments, while EfficientNet-B0 serves as an additional architecture comparison.

## Dataset

The experiments use the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/), with a selected subset of 10 breeds:

- **Cats:** Abyssinian, Bengal, Birman, Bombay, British Shorthair
- **Dogs:** Beagle, Boxer, Chihuahua, Pug, Samoyed

A total of **1,981 images** are divided using stratified sampling:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 1,386 | 70% |
| Validation | 297 | 15% |
| Testing | 298 | 15% |

The dataset is downloaded automatically through `torchvision.datasets.OxfordIIITPet` when the notebook is executed.

## Experimental Setup

All experiments use a consistent training pipeline with the following configuration:

| Configuration | Value |
|---|---|
| Input size | 224 × 224 |
| Batch size | 32 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss |
| Weight decay | 0.01 |
| Maximum epochs | 25 |
| Early stopping | Validation loss, patience 4 |
| Random seed | 42 |
| Normalization | ImageNet mean and standard deviation |

The pretrained models use ImageNet weights. Data augmentation is applied only to the training set, while validation and testing use deterministic preprocessing.

## Experiments

Six experimental conditions are evaluated:

- **B1 - Training from Scratch:** ResNet-18 initialized with random weights, with all parameters trainable.
- **B2 - Feature Extraction:** ImageNet-pretrained ResNet-18 with a frozen backbone and a newly trained classification head.
- **B3 - Progressive Fine-Tuning:** Pretrained ResNet-18 with `layer4` and `fc` unfrozen using differential learning rates.
- **B4 - Data Augmentation:** A controlled comparison of training with and without random augmentation using the selected transfer learning strategy.
- **B5 - EfficientNet-B0:** A pretrained architecture comparison using fine-tuning of the final feature stage and classifier.
- **C5 - Feature Map Comparison:** Visualization of ResNet-18 `layer4` activations before and after fine-tuning.

The augmentation pipeline includes random resized cropping, horizontal flipping, rotation, and brightness/contrast adjustments.

## Experimental Results

The following results were obtained from the final notebook execution.

| Experiment | Validation Accuracy | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|
| B1 - Scratch | 47.14% | 52.01% | 0.4997 |
| B2 - Feature Extraction | 96.97% | 96.98% | 0.9699 |
| B3 - Fine-Tuning | 98.32% | 96.64% | 0.9664 |
| B4 - Without Augmentation | 97.98% | 96.98% | 0.9696 |
| B4 - With Augmentation | 98.32% | 96.64% | 0.9664 |
| **B5 - EfficientNet-B0** | **98.99%** | **97.99%** | **0.9798** |

### Key Findings

**Transfer learning significantly outperformed training from scratch.** With only 1,386 training images, pretrained visual representations provided a substantial improvement in classification performance.

Feature extraction achieved strong results with only 5,130 trainable parameters, making it an effective option when training resources are limited.

Fine-tuning improved validation accuracy but did not consistently improve test performance. Data augmentation reduced the training-validation accuracy gap, although it did not improve ResNet-18 test accuracy in this experiment.

EfficientNet-B0 achieved the highest test Macro F1 of **0.9798**, using approximately 4.02 million total parameters, compared with approximately 11.18 million for ResNet-18.

## Feature Map Analysis

An additional experiment compares ResNet-18 `layer4` activations before and after fine-tuning.

The visualization uses three test images and displays the original input, pretrained feature activations, fine-tuned feature activations, and their differences.

The results show that fine-tuning changes the model's internal feature responses, although these changes do not guarantee correct predictions for visually similar breeds.

## Repository Structure

The project is organized as follows:

- `notebooks/Bagian_B_Eksperimen_Praktik.ipynb` — Complete executed notebook for B1–B5 and C5.
- `reports/Bagian_A_Analisis_Konseptual.pdf` — Conceptual analysis.
- `reports/Bagian_C_Laporan_Analisis.pdf` — Experimental results and discussion.
- `outputs/` — Training logs, configuration files, evaluation metrics, and figures.
- `requirements.txt` — Python dependencies required to run the notebook.

The main experimental output files include:

- `experiment_summary.csv` and `experiment_summary.json`
- `epoch_metrics.csv`
- `experiment_config.json`
- `misclassified_samples.csv`
- `C5_layer4_activation_comparison.csv`
- Training loss curves, confusion matrix, and feature map visualizations

## How to Run

### Google Colab (Recommended)

1. Open the notebook located in `notebooks/Bagian_B_Eksperimen_Praktik.ipynb` using Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Run all notebook cells in order.
4. Wait for the dataset download, training experiments, evaluation, and visualizations to finish.
5. Download the generated `Tugas05_BagianB_outputs.zip` archive containing the experiment outputs.

The notebook automatically downloads the Oxford-IIIT Pet dataset and pretrained model weights when required. Internet access is necessary for the initial downloads.

### Local Setup

Clone the repository using:

`git clone https://github.com/champsand/deep-learning-transfer-learning.git`

Install the required Python packages:

`pip install -r requirements.txt`

Open the notebook in Jupyter or another compatible notebook environment and execute its cells sequentially.

A CUDA-enabled GPU is recommended, particularly for training from scratch and fine-tuning. PyTorch installation may need to be adjusted for the local CUDA environment.

## References

- He et al. (2016). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- Tan & Le (2019). [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
- [PyTorch — Transfer Learning for Computer Vision Tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).
- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

## Notes

This repository contains the experiments and analysis for an individual Deep Learning coursework assignment. All reported performance metrics are based on the recorded notebook execution and the corresponding exported evaluation logs.
