This repository contains the implementation of a multimodal deep learning system that uses both image and audio inputs to classify bird species using a Siamese network architecture trained with triplet loss.

**📁 Project Structure**

fused_encoder.py: Defines the fused encoder model, combining VGG16-based image features with LSTM-based audio features (from YAMNet embeddings).

triplet_generator.py: Contains data loading and triplet generation logic for training. It processes image and audio files, extracts embeddings using YAMNet, and constructs triplets for Siamese training.

train.ipynb: Jupyter Notebook used to train the multimodal triplet-loss model, load triplet data, and monitor training metrics.

**🚀 Model Overview**

This system fuses two modalities:

Visual: Processed using a pre-trained VGG16 convolutional backbone (feature extraction with fine-tuning disabled for initial layers).

Audio: Pre-processed using librosa and YAMNet to extract temporal embeddings, then encoded via a bidirectional LSTM.

The fused output is normalized to form a 128-dimensional embedding vector.

This embedding is used to compute similarity between samples for triplet loss optimization, enabling robust discrimination between different bird species.

**🔁 Triplet Sampling**

The triplet_generator.py constructs triplets of the form (anchor, positive, hard negative):

Anchor & Positive: Same bird species (but different samples).

Hard Negative: Different species with high cosine similarity (to make training harder and better generalize embedding space).

The generator handles:

Image loading via OpenCV

Audio loading via soundfile, with resampling using resampy

YAMNet-based audio embedding extraction

Cosine similarity filtering for hard negative mining

**🧪 Training**

Training is performed using the notebook train.ipynb, where:

Each triplet is passed through the shared encoder.

Triplet loss is computed based on the distance between anchor-positive vs anchor-negative.

The model learns to bring similar samples closer and push different ones apart in embedding space.

**🛠 Requirements**

Python 3.7+

TensorFlow 2.x

OpenCV (cv2)

Librosa

SoundFile

Resampy

tqdm

scikit-learn

Install dependencies with:
