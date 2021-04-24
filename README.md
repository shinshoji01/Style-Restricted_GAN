# Style-Restricted_GAN
This repository is to introduce the implementation of our paper: [Style-Restricted GAN: Multi-ModalTranslation with Style Restriction UsingGenerative Adversarial Networks]().

---
## Introduction
This is the implementation of our model called Style-Restricted GAN (SRGAN), which is designed for the unpaired image translation with multiple styles. The main features of this models are 1) the enhancement of diversification and 2) the restriction of diversification. As for the former one, while the base model ([SingleGAN]()) employed KL divergence loss to restrict the distribution of encoded features like [VAE](), SRGAN exploits 3 new losses instead: batch KL divergence loss, correlation loss, and histogram imitation loss. When it comes to the restriction, in the previous, it wasn't explicitly designed to control how the generator diversifies the results, which can have an adverse effect on some applications. Therefore, in this paper, the encoder is pre-trained with the classification task before being used as an encoder.

We'll proceed this implementation in a notebook form. And we also share our docker environment in order for everybody to run the code as well as observing the implementation.

---
## Results
We would like to share our results first to briefly understand our objective. It consists of 2 experiments.

### Conventional KL Divergence Loss vs. Proposed Loss

<img src="./data/images/result_diversity_image.png" width="800">

### Style Restriction

<img src="./data/images/result_restriction_female.png" width="800">

---
## Notebooks
- 

---
## Docker

---
## Installation of some apps

**Git LFS (large file storage)**

Since this repository contains the parameters of the models. I used Git LFS to store a large file. The codes below are the recipe for this.

```bash
brew update
brew install git-lfs
```
- then, navigate to this repository.
```bash
git lfs install
git lfs fetch --all
git lfs pull
```

---
## Contact
Feel free to contact me if you have any questions (<s-inoue-tgz@eagle.sophia.ac.jp>).
