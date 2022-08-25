import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor, AutoFeatureExtractor
import torch
import torchaudio
import os

wav_path = '/data/A-VB/audio/wav/'
files = os.listdir(wav_path)

save_dir = "/data/A-VB/features/w2v2-lr"

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

# load model
processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust")

# head = [str(i) for i in range(768)]

# audio file is decoded on the fly
arrays = []
fs = 16000

for file in files[:4]:
    print(f"Processing {file}")

    array, fs = torchaudio.load(os.path.join(wav_path, file))
    # arrays.append(array)

    input = processor(array.squeeze(), sampling_rate=fs, return_tensors="pt")
# inputs = processor([array.squeeze() for array in arrays], sampling_rate=fs, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**input)

    last_hidden_states = outputs.last_hidden_state.squeeze().mean(axis=0).numpy()
    np.savetxt(f"{save_dir}/{str(file[:-3])}csv", last_hidden_states.reshape(1, -1), delimiter=',')
