# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pipe = pipeline("audio-classification", model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake", device=device)
res = pipe("call_vits_ft-Tower01_Antuona_02.wav")
print('fake:', res)

res = pipe(r"..\data\antona\Tower01_Antuona_02.wav")
print('real:', res)