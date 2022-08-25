import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch
import torchaudio
import os

wav_path = '/data/A-VB/audio/wav/'
files = os.listdir(wav_path)

save_dir = "/data/A-VB/features/w2v2-xlsr"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# load model
# Wav2Vec2 just extract the features from raw speech data. Then, there is no tokenizer has been defined for it. To load the processor you can use 
processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# head = [str(i) for i in range(768)]

# audio file is decoded on the fly
for file in files:
    print(f"Processing {file}")

    array, fs = torchaudio.load(os.path.join(wav_path, file))
    input = processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy()
    np.savetxt(f"{save_dir}/{str(file[:-3])}csv", last_hidden_states.reshape(1, -1), delimiter=',')
