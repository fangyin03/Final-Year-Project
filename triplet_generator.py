#triplet_generator.py
import os
import numpy as np
import cv2
import librosa
import resampy
import soundfile as sf
import tensorflow as tf
from tqdm import tqdm
from random import choice
from sklearn.metrics.pairwise import cosine_similarity
#import noisereduce as nr

IMAGE_SIZE = (224, 224)
AUDIO_SECONDS = 5
AUDIO_SR = 16000

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return np.zeros((*IMAGE_SIZE, 3))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)
    return img / 255.0

'''def denoise_audio(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)'''

def load_audio(audio_path, sr=AUDIO_SR):
    audio_data, file_sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    if file_sr != sr:
        audio_data = resampy.resample(audio_data, file_sr, sr)
    if len(audio_data) < sr * AUDIO_SECONDS:
        audio_data = np.pad(audio_data, (0, sr * AUDIO_SECONDS - len(audio_data)))
    else:
        audio_data = audio_data[:sr * AUDIO_SECONDS]
    return audio_data.astype(np.float32)

def extract_yamnet_sequence(yamnet_model, audio_data):
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    embeddings = yamnet_model(audio_tensor)[1]
    return embeddings.numpy()  # [frames, 1024]

def prepare_triplets(image_dir, audio_dir, yamnet_model, num_triplets_per_class=100):
    triplets = []
    species_list = sorted([
        s for s in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, s)) and not s.startswith('.')
    ])

    for species in tqdm(species_list, desc="Preparing Triplets"):
        img_paths = [os.path.join(image_dir, species, f) for f in os.listdir(os.path.join(image_dir, species)) if f.endswith(('.jpg', '.png'))]
        aud_paths = [os.path.join(audio_dir, species, f) for f in os.listdir(os.path.join(audio_dir, species)) if f.endswith('.wav')]

        if len(img_paths) < 2 or len(aud_paths) < 2:
            continue

        for _ in range(num_triplets_per_class):
            # Anchor
            a_img_path = choice(img_paths)
            a_audio_data = load_audio(choice(aud_paths))
            a_emb_seq = extract_yamnet_sequence(yamnet_model, a_audio_data)

            # Positive
            p_img_path = choice(img_paths)
            p_audio_data = load_audio(choice(aud_paths))
            p_emb_seq = extract_yamnet_sequence(yamnet_model, p_audio_data)

            # Hard Negative
            neg_species = choice([s for s in species_list if s != species])
            neg_img_path = choice([
                os.path.join(image_dir, neg_species, f) for f in os.listdir(os.path.join(image_dir, neg_species)) if f.endswith(('.jpg', '.png'))
            ])
            neg_aud_path = choice([
                os.path.join(audio_dir, neg_species, f) for f in os.listdir(os.path.join(audio_dir, neg_species)) if f.endswith('.wav')
            ])
            n_audio_data = load_audio(neg_aud_path)
            n_emb_seq = extract_yamnet_sequence(yamnet_model, n_audio_data)

            if cosine_similarity(a_emb_seq.mean(axis=0).reshape(1, -1), n_emb_seq.mean(axis=0).reshape(1, -1))[0][0] < 0.6:
                continue

            triplets.append((a_img_path, a_emb_seq, p_img_path, p_emb_seq, neg_img_path, n_emb_seq))

    return triplets
