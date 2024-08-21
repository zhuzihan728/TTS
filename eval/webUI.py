import json
import gradio as gr
import os
from Levenshtein import distance as levenshtein_distance
from utils import normalized_text_to_phonemes
from api import to_wave, call_vits_ft, TTS_FNs
from asr import load_asr_model, transcribe_audio
import ddc
import spk_verify
ASR_MODEL = None
asr_model = None
asr_processor = None
# Placeholder for paths
generated_audio_paths = []
ground_truth_audio_paths = []
transcriptions = []
history = []
print(TTS_FNs)
def calculate_phoneme_error_rate(asr_text, ground_truth_text):
    asr_phonemes = normalized_text_to_phonemes(asr_text)
    ground_truth_phonemes = normalized_text_to_phonemes(ground_truth_text)
    print(f'{asr_text} -> {asr_phonemes}')
    print(f'{ground_truth_text} -> {ground_truth_phonemes}')
    edit_distance = levenshtein_distance(asr_phonemes, ground_truth_phonemes)
    per = edit_distance / len(ground_truth_phonemes)
    print(f'PER: {per:.4f}')
    return per

def calculate_per(asr_model_name):
    global ASR_MODEL, asr_model
    if ASR_MODEL is None or ASR_MODEL != asr_model_name:
        ASR_MODEL = asr_model_name
        asr_model = load_asr_model(asr_model_name)
    model = asr_model
    
    per_list = []
    
    for i in range(len(ground_truth_audio_paths)):
        # ground_truth_transcription = transcriptions[i]
        ground_truth_asr = transcriptions[i]
        generated_asr = transcribe_audio(generated_audio_paths[i], model)
        per_list.append(calculate_phoneme_error_rate(generated_asr, ground_truth_asr))
    
    avg_per = sum(per_list) / len(per_list)
    
    return avg_per

def calulate_ddc():
    ddc_scores = ddc.get_avg_ddc(generated_audio_paths)
    print([i for i in zip(generated_audio_paths, ddc_scores)])
    avg_ddc = sum(ddc_scores) / len(ddc_scores)
    return avg_ddc

def calculate_svs():
    speaker_verification_scores = []
    for i in range(0, len(ground_truth_audio_paths)):
        speaker_verification_scores.append(spk_verify.speaker_verification(ground_truth_audio_paths[i], generated_audio_paths[i]))
    avg_svs = sum(speaker_verification_scores) / len(speaker_verification_scores)
    return avg_svs

def calculate_metrics(asr_model_name, tts_function_name, json_path):
    per = calculate_per(asr_model_name)
    ddc = calulate_ddc()
    svs = calculate_svs()
    history.append([tts_function_name, per, ddc, svs, asr_model_name, os.path.basename(json_path.name)])
    return f"{tts_function_name} with {asr_model_name} ASR model: PER={per:.4f}, DDC={ddc:.4f}, SVS={svs:.4f}", history


def load_json(json_file):
    global ground_truth_audio_paths, transcriptions, generated_audio_paths
    
    with open(json_file.name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    ground_truth_audio_paths = [item['audio_path'] for item in data]
    transcriptions = [item['transcription'] for item in data]
    generated_audio_paths = [item['audio_path'] for item in data]
    return f"Loaded {len(ground_truth_audio_paths)} items from JSON."

def generate_audio_from_json(tts_function_name):
    global generated_audio_paths, transcriptions, ground_truth_audio_paths
    if tts_function_name == 'ground truth':
        generated_audio_paths = ground_truth_audio_paths[:]
        return f"Using {len(generated_audio_paths)} ground truth audios."
    generated_audio_paths.clear()
    tts_function = TTS_FNs[tts_function_name]
    for i, text in enumerate(transcriptions):
        audio, rate = tts_function(text)
        output_path = f'{tts_function_name}-' + os.path.basename(ground_truth_audio_paths[i])
        to_wave(audio, rate, output_path)
        generated_audio_paths.append(output_path)
    
    return f"Generated {len(generated_audio_paths)} audios from transcriptions in JSON."
# Define Gradio UI components
def create_gradio_ui():
    with gr.Blocks() as demo:
        json_input = gr.File(label="Upload JSON File", type='filepath')
        tts_function_input = gr.Dropdown(label="Select TTS Function", choices=list(TTS_FNs.keys()))
        asr_model_input = gr.Dropdown(label="Enter ASR Whisper Size", choices=["small", "medium", "large"])
        generate_button = gr.Button("Generate Audio from JSON")
        metrics_button = gr.Button("Calculate Metrics")
        output_display = gr.Textbox(label="Output")
        history_display = gr.Dataframe(label="History", headers=['TTS model', 'PER', 'DDC', 'SVS', 'ASR model', 'JSON path'], datatype=['str', 'number', 'number', 'number', 'str', 'str'])
        

        # Set up interactions
        json_input.change(load_json, inputs=json_input, outputs=output_display)
        generate_button.click(generate_audio_from_json, inputs=tts_function_input, outputs=output_display)
        metrics_button.click(calculate_metrics, inputs=[asr_model_input, tts_function_input, json_input], outputs=[output_display, history_display])
    return demo

# Run the Gradio app
gr_interface = create_gradio_ui()
gr_interface.launch()
