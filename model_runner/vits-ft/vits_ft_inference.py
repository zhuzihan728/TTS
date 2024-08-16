from pathlib import Path
import utils
from models import SynthesizerTrn
import torch
from torch import no_grad, LongTensor
import librosa
from text import text_to_sequence, _clean_text
import commons
import scipy.io.wavfile as wavf
import os
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

#========= configs =========#
model_path = "vits-ft/G_latest.pth"
config_path = "vits-ft/modified_finetune_speaker.json"
output_path = "res_output"
output_dir = Path(output_path)
output_dir.mkdir(parents=True, exist_ok=True)
language = "简体中文"
text = "你好，我是琪亚娜！"
spk = "kyn"
noise_scale = 0.667
noise_scale_w = 0.6
length = 1
output_name = "output"

#========= load model =========#
hps = utils.get_hparams_from_file(config_path)
net_g = SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)
_ = net_g.eval()
_ = utils.load_checkpoint(model_path, net_g, None)

speaker_ids = hps.speakers
print("模型加载完毕！")

#========= inference =========#
def inference(text):
    if language is not None:
        text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[spk]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                length_scale=1.0 / length)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return audio, hps.data.sampling_rate
    return None, hps.data.sampling_rate
# def inference2wav(text, output_name = None):
#     if not output_name:
#         output_name = len(os.listdir(output_dir))
#         output_name = str(output_name)
    
#     audio, _ = inference(text)
#     if audio is not None:
#         wavf.write(str(output_dir)+"/"+output_name+".wav",hps.data.sampling_rate,audio)

# if __name__ == "__main__":
#     input_text = ""
#     while True:
#         input_text = input("请输入文本：")
#         if input_text == "exit":
#             break
#         inference2wav(input_text)


    
    