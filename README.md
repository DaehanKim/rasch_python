# Rasch Model for Python

### Note

This model implements Rasch model using torch library.
Using (student, skill, response) triplets appearing on the dataset, Rasch model is trained using backpropagation on the negative log likelihood.  

### Requirements

- torch
- visdom
- tqdm

### How to Run

```
python rasch_main.py
```