# Style-Restricted Generative Adversarial Networks
This repository is to introduce the implementation of our paper: [Style-Restricted GAN: Multi-ModalTranslation with Style Restriction Using Generative Adversarial Networks](https://arxiv.org/abs/2105.07621).

---
## Introduction
This is the implementation of our model called Style-Restricted GAN (SRGAN), which is designed for the unpaired image translation with multiple styles. The main features of this models are 1) the enhancement of diversification and 2) the restriction of diversification. As for the former one, while the base model ([SingleGAN](https://github.com/Xiaoming-Yu/SingleGAN)) employed KL divergence loss to restrict the distribution of encoded features like [VAE](https://arxiv.org/abs/1312.6114), SRGAN exploits 3 new losses instead: batch KL divergence loss, correlation loss, and histogram imitation loss. When it comes to the restriction, in the previous, it wasn't explicitly designed to control how the generator diversifies the results, which can have an adverse effect on some applications. Therefore, in this paper, the encoder is pre-trained with the classification task before being used as an encoder.

We'll proceed this implementation in a notebook form. And we also share our docker environment in order for everybody to run the code as well as observing the implementation.

---
## Results
We would like to share our results first to briefly understand our objective. It consists of 2 experiments.

### Conventional KL Divergence Loss vs. Proposed Loss

<img src="./data/images/result_diversity_image.png" width="800">

### Style Restriction

<img src="./data/images/result_restriction_female.png" width="800">

https://user-images.githubusercontent.com/28431328/119615142-09661e00-be3a-11eb-9115-655bbc16a4e3.mp4

---
## Notebooks
- `01-test_Conventional_SingleGAN.ipynb`
  - examination of the conventional SingleGAN
- `01-train_Conventional_SingleGAN.ipynb`
  - training of the conventional SingleGAN
- `02-test_SingleGAN_soloD.ipynb`
  - examination of the SingleGAN with a solo discriminator
- `02-train_SingleGAN_soloD.ipynb`
  - training of the SingleGAN with a solo discriminator
- `03-test_Style-Restricted_GAN_nopretraining.ipynb`
  - examinaton of Style-Restricted GAN without pretraining
- `03-train_Style-Restricted_GAN_nopretraining.ipynb`
  - training of Style-Restricted GAN without pretraining
- `04_Facial_Recognition-Encoder.ipynb`
  - classification for SRGAN
- `05-test_Style-Restricted_GAN.ipynb`
  - examination of Style-Restricted GAN
- `05-train_Style-Restricted_GAN.ipynb`
  - training of Style-Restricted GAN
- `06_Comparison_PRDC.ipynb`
  - compare all the models
- `A_CelebA_dataset_usage.ipynb`
  - How to download and use CelebA dataset
- `B_Facial_Recognition-VGG_Model.ipynb`
  - classification for evaluation metrics

---
## Docker
In this repository, we share the environment that you can run the notebooks.
1. Build the docker environment.
    - with GPU
      - `docker build --no-cache -f Docker/Dockerfile.gpu .`
    - without GPU
      - `docker build --no-cache -f Docker/Dockerfile.cpu .`
2. Check the \<IMAGE ID\> of the created image.
    - `docker images`
3. Run the docker environment
    - with GPU
      - `docker run --rm --gpus all -it -p 8080:8080 -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -v ~/:/work <IMAGE ID> bash`
    - without GPU
      - `docker run --rm -it -p 8080:8080 -e LOCAL_UID=$(id -u $USER) -e LOCAL_GID=$(id -g $USER) -v ~/:/work <IMAGE ID> bash`
4. Run the jupyter lab
    - `nohup jupyter lab --ip=0.0.0.0 --no-browser --allow-root --port 8080 --NotebookApp.token='' > nohup.out &`
5. Open the jupyter lab
    - Put http://localhost:8080/lab? to web browser.

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
## Coming soon
Some are not explained which include:
- explanations of some functions and models.

---
## Contact
Feel free to contact me if you have any questions (<s-inoue-tgz@eagle.sophia.ac.jp>).
