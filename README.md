# Facial Recognition

In this project, I created my own [facial-recognition model](https://huggingface.co/jamesnatulan/face-recognition/resolve/main/face-recog-resnet18.pt?download=true) using a Siamese Twin Network. The model that I've used as the twin network is a ResNet. The loss function I've used is a Contrastive Loss function that uses Euclidean distance as the distance metric. I've trained using the provided train and test splits from the [LFW dataset](https://vis-www.cs.umass.edu/lfw/). I've trained on my own machine that has an NVIDIA 3080Ti with 12GB or VRAM. The model is available to download from Huggingface on the link provided below:

[face-recog-resnet18.pt](https://huggingface.co/jamesnatulan/face-recognition/resolve/main/face-recog-resnet18.pt?download=true)


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

## Demo

![demo](assets/docs/demo.GIF)


To run the demo on your own set of faces, provide the path to the faces you want to use (edit the `demo.py` file and add your own paths to the constants at the top). The directory must follow the structure of the sample faces I provided in this project, which is:
- FACES DIRECTORY
    - Name of person 1
        - image 1.jpg
        - image 2.jpg
        ..
        - image n.jpg
    ..
    - Name of person n

You can add any number of person, and any number of images that represents them. Make sure you have the `yolon-face.pt` model available, as it is used for face detection. You can get the model from [this repository](https://github.com/akanametov/yolo-face), or download it from my HuggingFace repo [here](https://huggingface.co/jamesnatulan/face-recognition/blob/main/yolov8n-face.pt). 

To run the demo:

```bash
uv run demo.py
```


## Implementing facial-recognition from scratch

To try it on your own, run the following scripts:

```bash
uv run scripts/train.py
uv run scripts/test.py
```

You can tweak some of the parameters by going to the scripts, and modifying the constants found at the top. On my own run, I've produced the following results by training with the following configuration:

### Training Parameters

|  Parameter | Value |
| -------- | ----- |
| Model |  ResNet18 |
| Batch Size | 64 |
| Number of Epochs | 50 |
| Learning Rate | 5e-5 |
| Optimizer | Adam |
| Weight Decay | 5e-4 |


### Metrics

The training loss and validation loss across the training run are shown in the image below:


![loss curve](assets/docs/loss_curve.jpg)


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


## Citations

Labeled Faces in the Wild

```BibTeX
@TechReport{LFWTech,
  author =       {Gary B. Huang and Manu Ramesh and Tamara Berg and 
                  Erik Learned-Miller},
  title =        {Labeled Faces in the Wild: A Database for Studying 
                  Face Recognition in Unconstrained Environments},
  institution =  {University of Massachusetts, Amherst},
  year =         2007,
  number =       {07-49},
  month =        {October}}
```

Contrastive Loss
```BibTex
@INPROCEEDINGS{1467314,
  author={Chopra, S. and Hadsell, R. and LeCun, Y.},
  booktitle={2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05)}, 
  title={Learning a similarity metric discriminatively, with application to face verification}, 
  year={2005},
  volume={1},
  number={},
  pages={539-546 vol. 1},
  keywords={Character generation;Drives;Robustness;System testing;Spatial databases;Glass;Artificial neural networks;Support vector machines;Support vector machine classification;Face recognition},
  doi={10.1109/CVPR.2005.202}}
```

YOLO Face Detection models are from this repository: https://github.com/akanametov/yolo-face

```BiBTex
@software{Jocher_YOLO_by_Ultralytics_2023,
author = {Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
license = {GPL-3.0},
month = jan,
title = {{YOLO by Ultralytics}},
url = {https://github.com/ultralytics/ultralytics},
version = {8.0.0},
year = {2023}
}
```

