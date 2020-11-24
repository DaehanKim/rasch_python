# Rasch Model for Python

This repository implements Rasch model using torch library.
Using (student, skill, response) triplets appearing on the assistment dataset, Rasch model is trained using backpropagation on the negative [log joint likelihood](https://en.wikipedia.org/wiki/Rasch_model_estimation). 

Dataset attached is made from ASSISTment dataset using both skill-buider and non-skill-builder with samples having at least one skill tags. First row represents student id, second skill sequence, third correctness of response. Dataset is splited into train and test, ensuring all student lying in test set exist in train set. Train : Test ratio is 3:1 where sequences with less than 4 items were put into train set.

### Requirements

- torch
- visdom
- tqdm

### How to Run

```
python -m visdom.server
python rasch_main.py
```