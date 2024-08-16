import whisper
import soundfile as sf
import torch
def load_asr_model(model_size):
    # load model and processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model(model_size, device=device)
    return model

def transcribe_audio(audio_path, model):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result.text
    
   

if __name__ == "__main__":
    asr_model_name = "medium"
    audio_path = "../data/antona/Tower01_Antuona_02.wav"
    model = load_asr_model(asr_model_name)
    transcription = transcribe_audio(audio_path, model)
    print(transcription)