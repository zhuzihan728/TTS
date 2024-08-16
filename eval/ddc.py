# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# tried HyperMoon/wav2vec2-base-960h-finetuned-deepfake; abhishtagatya/wav2vec2-base-960h-itw-deepfake; Hemg/small-deepfake; motheecreator/Deepfake-audio-detection;MelodyMachine/Deepfake-audio-detection-V2 
pipe = pipeline("audio-classification", model="DavidCombei/wavLM-base-DeepFake_UTCN", device=device)

def get_avg_ddc(paths):
    res = pipe(paths)
    print(res)
    res = [i[0]['score'] if i[0]['label'] == 'LABEL_1' else i[1]['score'] for i in res]
    return res

if __name__ == "__main__":
    res = pipe(["call_vits_ft-Tower01_Antuona_02.wav", "call_vits_ft-Tower04_Antuona_04_02.wav", "call_vits_ft-Tower03_Antuona_04.wav"])
    print('fake:', res)

    res = pipe([r"..\data\antona\Tower01_Antuona_02.wav", r"..\data\antona\Tower04_Antuona_04_02.wav", r"..\data\antona\Tower03_Antuona_04.wav"])
    print('real:', res)
    
    res = get_avg_ddc([r"..\data\antona\Tower01_Antuona_02.wav", r"..\data\antona\Tower04_Antuona_04_02.wav", "call_vits_ft-Tower03_Antuona_04.wav"])
    print('avg:', res)