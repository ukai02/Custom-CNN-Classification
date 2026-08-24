# Custom CNN Architecture for Image Classification

An end-to-end implementation of a Convolutional Neural Network (CNN) built entirely from scratch using PyTorch. This project was developed as part of a competitive academic assignment (CS771) to perform multi-class image classification (16 classes) on a complex dataset featuring three complementary audio-to-image representations.

##  Performance
* **Private Leaderboard Score (Macro F1):** 93.92%
* **Public Leaderboard Score (Macro F1):** 93.67%

*(Evaluated on the Kaggle competition leaderboard, utilizing 30% of test data for public scoring and 70% for private scoring).*

##  Key Features & Mathematical Implementations
Instead of relying on standard high-level PyTorch abstractions (like `nn.Conv2d` or `nn.BatchNorm2d`), the core neural network layers were engineered from low-level mathematical principles:

* **Custom Convolutions:** Implemented forward passes using tensor unfolding and matrix multiplication (`matmul`) for optimized patch processing.
* **Batch Normalization:** Manually tracked running means and variances with custom momentum updates during the training and inference phases.
* **Pooling Layers:** Engineered custom `max_pool` and `adaptive_avg_pool` functions to dynamically handle variable spatial dimensions.
* **Custom Activations & Regularization:** Built mathematically exact representations of Leaky ReLU, standard Dropout, and Spatial Dropout.

##  Model Architecture
The network processes a concatenated 3-channel input (composed of Mel-spectrogram, CQT, and Chromagram representations) through a progressively expanding channel architecture:
* **Block 1:** 3 ➔ 28 Channels (with Spatial Dropout)
* **Block 2:** 28 ➔ 52 Channels
* **Block 3:** 52 ➔ 96 Channels
* **Block 4:** 96 ➔ 144 Channels
* **Block 5:** 144 ➔ 200 Channels
* **Fully Connected Head:** 200 ➔ 128 ➔ 16 Classes (Target)

##  Advanced Data Augmentation
To prevent overfitting and maximize the extraction of complementary audio-image characteristics, a highly robust multi-input augmentation pipeline was designed:
* **SpecAugment:** Applied custom time and frequency masking to input tensors.
* **Random Time Shifting:** Applied circular shifts along the time axis to simulate temporal invariance.
* **Chroma Augmentation:** Specifically tailored time-masking logic applied strictly to the chromagram input representations.

##  Tech Stack
* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Data Processing & Metrics:** Pandas, NumPy, scikit-learn
* **Visualization:** Matplotlib, Seaborn, tqdm

##  Usage
To run the model and view the mathematical implementations, open the notebook:
```bash
jupyter notebook final-arch.ipynb
