# Sign Language Capstone Project
### Master of Science in Data Analytics
### George Washington University

## Why & What
Although there are many language translation applications in the market, very few, involve sign language. People who use sign language may have hearing or speaking problems or are in the deaf community. They seem like a minority group but have a 0.5 billion population worldwide surfing from hearing disability, not to mention if we include their relatives, friends, and coworkers. It is quite a potential market, but few people pay attention to it.

Some sign language translation applications are in the market, but they are limited to word-to-word translation. Seeing the need of this situation, we want to try to offer a more straightforward solution for sign language translation by machine learning algorithms.

## Data

How2Sign is an open-source multimodal and multiview continuous American Sign Language dataset with annotations. All the videos have sentence-level alignment.

- 80 hours of sign language videos and corresponding English transcriptions

- 31 GB training, 1.7 GB validation, 2 GB test

Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier. 2021. *How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language*. Conference on Computer Vision and Pattern Recognition (CVPR). https://how2sign.github.io  

## Tools we used:

- Mediapipe & OpenCV — to process the video data

- PyTorch — to build the model

- EC2, AWS — to train the model

## Workflow

- Video-Processing and Transformation

- Text Data Transformation

- Labeling

- Modeling & Training

- Evaluation

