import json
import gradio as gr
import os
from Levenshtein import distance as levenshtein_distance
from utils import normalized_text_to_phonemes
from api import to_wave, call_vits_ft, TTS_FNs
from asr import load_asr_model, transcribe_audio
import ddc
ASR_MODEL = None
asr_model = None
asr_processor = None
# Placeholder for paths
generated_audio_paths = []
ground_truth_audio_paths = []
transcriptions = []
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
        ground_truth_asr = transcribe_audio(ground_truth_audio_paths[i], model)
        generated_asr = transcribe_audio(generated_audio_paths[i], model)
        per_list.append(calculate_phoneme_error_rate(generated_asr, ground_truth_asr))
    
    avg_per = sum(per_list) / len(per_list)
    
    return f"Average PER for {len(per_list)} items: {avg_per:.4f}"

def calulate_ddc():
    ddc_scores = ddc.get_avg_ddc(generated_audio_paths)
    print([i for i in zip(generated_audio_paths, ddc_scores)])
    avg_ddc = sum(ddc_scores) / len(ddc_scores)
    return f"Average DDC for {len(ddc_scores)} items: {avg_ddc:.4f}"

def load_json(json_file):
    global ground_truth_audio_paths, transcriptions
    
    with open(json_file.name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    ground_truth_audio_paths = [item['audio_path'] for item in data]
    transcriptions = [item['transcription'] for item in data]
    return f"Loaded {len(ground_truth_audio_paths)} items from JSON."

def generate_audio_from_json(tts_function_name):
    global generated_audio_paths, transcriptions, ground_truth_audio_paths
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
        PER_button = gr.Button("Calculate PER")
        DDC_button = gr.Button("Calculate DDC")
        output_display = gr.Textbox(label="Output")

        # Set up interactions
        json_input.change(load_json, inputs=json_input, outputs=output_display)
        generate_button.click(generate_audio_from_json, inputs=tts_function_input, outputs=output_display)
        PER_button.click(calculate_per, inputs=asr_model_input, outputs=output_display)
        DDC_button.click(calulate_ddc, outputs=output_display)
    return demo

# Run the Gradio app
gr_interface = create_gradio_ui()
gr_interface.launch()
