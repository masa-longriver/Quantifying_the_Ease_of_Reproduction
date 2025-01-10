# Quantifying the Ease of Reproducing Training Data in Unconditional Diffusion Models

This is an official code for the paper "Quantifying the Ease of Reproducing Training Data in Unconditional Diffusion Models" 

by Masaya Hasegawa and Koji Yasuda.

This paper is accepted at [the 1st Workshop on Preparing Good Data for Generative AI: Challenges and Approaches](https://sites.google.com/servicenow.com/good-data-2025/), 

which is a workshop of the 39th Annual Conference on Artificial Intelligence (AAAI).

---

## How to run the code

### Dependencies

Run the following command to install the dependencies:

```bash
pip install -r requirements.txt
```

### 1. Training the Diffusion Model

Run the following command to train the unconditional diffusion model:

```bash
cd src/scripts
python train_model.py --dataset=<dataset_name> --result_dir_name=<result_dir_name(optional)>
```

You can use the following datasets:
- `cifar10`
- `cifar10_2^7` (Sampled 2^7 images from CIFAR-10)
- `celeba`

### 2. Sampling Images from the Trained Model

Run the following command to sample images from the trained model:

```bash
cd src/scripts
python sample_images.py --dataset=<dataset_name> --result_dir_name=<result_dir_name(optional)>
```

### 3. Evaluating the Ease of Reproduction

Run the following command to evaluate the ease of reproduction:

```bash
cd src/scripts
python evaluate_ease_of_reproduction.py --dataset=<dataset_name> --result_dir_name=<result_dir_name(optional)>
```

### Parameters Setting

You can set the model parameters in the `config/<dataset_name>.yaml` file.

You can also set the dataset parameters in the `config/<dataset_name>.yaml` file.


## References

This work is built upon the following paper:

- Song, Y.; Sohl-Dickstein, J.; Kingma, D.P.; Kumar, A.; Ermon, S.; Poole, B. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. Paper represented at the 9th International Conference on Learning Representations (ICLR).
