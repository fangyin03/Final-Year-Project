This repository contains the implementation of a cross-modal dual-stream network that uses both image and audio inputs to recognise bird species using network architecture trained with triplet loss.

**📁 Project Structure**

**fused_encoder.py:** Defines the fused encoder model, combining VGG16-based image features with LSTM-based audio features (from YAMNet embeddings).

**triplet_generator.py:** Contains data loading and triplet generation logic for training. It processes image and audio files, extracts embeddings using YAMNet, and constructs triplets for dual-stream network training.

**train.ipynb:** Jupyter Notebook used to train the cross-modal triplet-loss model, load triplet data, and monitor training metrics.

**🚀 Model Overview**

This system fuses two modalities:

**Visual:** Processed using a pre-trained VGG16 convolutional backbone (feature extraction with fine-tuning disabled for initial layers).

**Audio:** Pre-processed using librosa and YAMNet to extract temporal embeddings, then encoded via a bidirectional LSTM.

-The fused output is normalized to form a 128-dimensional embedding vector.

-This embedding is used to compute similarity between samples for triplet loss optimization, enabling robust discrimination between different bird species.

**🔁 Triplet Sampling**

The triplet_generator.py constructs triplets of the form (anchor, positive, hard negative):

**Anchor & Positive:** Same bird species (but different samples).

**Hard Negative:** Different species with high cosine similarity (to make training harder and better generalize embedding space).

The generator handles:

-Image loading via OpenCV

-Audio loading via soundfile, with resampling using resampy

-YAMNet-based audio embedding extraction

-Cosine similarity filtering for hard negative mining

**🧪 Training**

Training is performed using the notebook train.ipynb, where:

-Each triplet is passed through the shared encoder.

-Triplet loss is computed based on the distance between anchor-positive vs anchor-negative.

-The model learns to bring similar samples closer and push different ones apart in embedding space.

**🛠 Requirements**

-Python 3.7+

-TensorFlow 2.x

-Numpy

-OpenCV (cv2)

-Librosa

-SoundFile

-Resampy

-tqdm

-Matplotlib

-scikit-learn

-random

<pre>├── image_dir/
│   ├── BirdSpeciesA/
│   │   ├── img1.jpg
│   │   └── ...
│   └── BirdSpeciesB/
│       └── ...
├── audio_dir/
│   ├── BirdSpeciesA/
│   │   ├── audio1.wav
│   │   └── ...
│   └── BirdSpeciesB/
│       └── ...</pre>

**📦 How to Run**

Before starting to build the model, if you don’t have Python installed or want an easier way to manage packages and environments, you can install Anaconda Navigator: (Recommended for Managing Dependencies)

👉 Download and install it from the official site:
https://www.anaconda.com/products/distribution

After installation, you can launch Jupyter Notebook and manage environments directly from the Anaconda Navigator GUI.

**1. 🔧 Install Dependencies**

First, install all required Python packages:
<pre>pip install tensorflow numpy opencv-python librosa soundfile resampy tqdm matplotlib scikit-learn
</pre>

**2. 🧠 Load and Prepare Triplets**

To generate training data triplets using your dataset, run the following inside a Python script or interactive session:
<pre>from triplet_generator import prepare_triplets
import tensorflow_hub as hub

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
triplets = prepare_triplets("train_images", "train_audio", yamnet_model)
</pre>

This will return a list of (anchor_img_path, anchor_audio_seq, positive_img_path, ..., negative_img_path, ...) triplets.

**3. 🏗️ Build the Fused Encoder Model**

<pre>from fused_encoder import create_fused_encoder

model = create_fused_encoder()
model.summary()
</pre>

**4. 📚 Train the Model**

Open the train.ipynb notebook and run the cells sequentially. The notebook demonstrates:

1. Loading the generated triplets

2. Preparing the training batches

3. Building the triplet loss function

4. Training the fused encoder using model.fit(...)

**📌 Notes**

This system is ideal for bird call recognition combined with visual cues, especially in noisy environments or incomplete modality conditions.

For better accuracy, ensure balanced and sufficient samples per class for both modalities.

The fused encoder can be exported and reused for embedding-based retrieval or classification.

**📜 License**

MIT License © 2025
