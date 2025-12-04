# Beyond Pixels: Disentangled Beta-VAE on Japanese Calligraphy

This repository contains a tutorial and implementation of a **Disentangled Variational Autoencoder ($\beta$-VAE)** applied to the **Kuzushiji-MNIST (KMNIST)** dataset.

Unlike standard MNIST (digits), KMNIST consists of ancient Japanese cursive calligraphy. This project demonstrates how introducing a weighting parameter ($\beta$) to the KL-Divergence term allows the model to learn the *geometry* of handwriting strokes rather than just memorizing pixels.

## 🎨 Visual Results
The model learns a continuous latent manifold, allowing us to "morph" one Japanese character into another. This proves the model understands the structural stroke data.

![Latent Space Morphing](morphing_result.png)
*(Figure: Latent interpolation between two distinct character classes)*

## 📂 Project Structure
* **`Disentangled_VAE_KMNIST.ipynb`**: The complete, runnable Jupyter Notebook. It includes:
    * Data loading (via TensorFlow Datasets).
    * Custom `DisentangledVAE` class with overridden `train_step`.
    * Latent space visualization and interpolation experiments.
* **`requirements.txt`**: List of dependencies required to run the code.

## 🚀 Key Technical Features
### 1. The Beta-VAE Loss Function
We modify the standard Evidence Lower Bound (ELBO) loss function to aggressively enforce the Gaussian prior.
The formula used is:

$$\mathcal{L} = \mathcal{L}_{recon} + \beta \cdot D_{KL}(q(z|x) || p(z))$$

By setting **$\beta = 4.0$**, we force the model to disentangle the latent factors of variation.

### 2. The Reparameterization Trick
To allow backpropagation through a stochastic sampling node, we implement a custom Keras layer that samples $\epsilon \sim \mathcal{N}(0, 1)$ and computes:
$$z = \mu + e^{0.5\sigma} \cdot \epsilon$$

### 3. Latent Arithmetic
The project demonstrates that the learned latent space is continuous. By linearly interpolating between the latent vectors of two different characters, we generate smooth morphological transitions that do not exist in the training set.

## ⚙️ How to Run
1.  Open the `.ipynb` file in **Google Colab** (Recommended for free GPU access).
2.  Ensure the Runtime type is set to **T4 GPU**.
3.  Run all cells to train the model and generate the visualizations.

## 🛠 Dependencies
* Python 3.x
* TensorFlow 2.x
* TensorFlow Datasets (TFDS)
* NumPy
* Matplotlib

## 📜 License
This project is open-source and available under the MIT License.

## 🔗 Acknowledgements
* **Dataset**: [Kuzushiji-MNIST](https://github.com/rois-codh/kmnist) (Clanuwat et al., 2018).
* **Concept**: *Beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework* (Higgins et al., 2017).
