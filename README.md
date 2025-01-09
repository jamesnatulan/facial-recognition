# Facial Recognition
WRITE HERE INTRODUCTION TO THE PROJECT

## Setup

This repository uses [UV](https://astral.sh/blog/uv) as its python dependency management tool. Install UV by:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Initialize virtual env and activate
```bash
uv venv
source .venv/bin/activate
```

Install dependencies with:
```bash
uv sync
```

## Implementing facial-recognition from scratch

I have attempted to implement and create my own facial-recognition model using a Siamese Twin Network. The model that I've used as the twin network is a ResNet. The loss function I've used is a Contrastive Loss function. I've trained using the provided train and test splits from the LFW dataset. I've trained on my own machine that has an NVIDIA 3080Ti with 12GB or VRAM. 

To try it on your own, run the following scripts:
```bash
uv run scripts/train.py
uv run scripts/test.py
```

You can tweak some of the parameters by going to the scripts, and modifying the constants found at the top. On my own run, I've produced the following results by training with the following configuration:

### Training Parameters
|  Parameter | Value |
| -------- | ----- |
| Accuracy |  0.645 |
| Precision | 0.677 |
| Recall | 0.554 |
| F1 Score | 0.609 |

### Metrics
The training loss and validation loss across the training run are shown in the image below:


![loss curve](assets/docs/loss_curve.png)


I measured the model performance using the 4 metrics below. Note that the input images are considered match if the euclidean distance between them are less than 0.5. The true positives, true negatives, false positives, and false negatives are then decided by this. True positives for when the input are match, and the label says they are match, and so on. Also shown below is the metrics for the base ResNet18 model, for comparison.

|  Metric | Finetuned Model | Base Model |
| -------- | ----- | ----- |
| Accuracy |  0.645 | 0.5 |
| Precision | 0.677 | Undefined |
| Recall | 0.554 | 0.0 |
| F1 Score | 0.609 | Undefined |


A quick glance shows that the model didn't really show state-of-the-art results, but it showed improvements over the base ResNet model. To compete with SOTA, one should have a huge amount of data to train on and a lot of compute power to run the training. For reference, the table below shows the SOTA models, and the amount of training images they are trained on.

|  Model | Training Images |
| -------- | ----- |
| DeepFace |  0.5M |
| DeepID2 | 0.2M |
| VGG-Face | 2.6M |
| DCMN | 0.5M |


## Using Deepface Library

To use the SOTA models, I used the [Deepface library](https://github.com/serengil/deepface) which is a lightweight Face Recognition library in Python. It also supports Facial Attribute Analysis such as age, gender, emotion, and race. 

