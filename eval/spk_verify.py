from modelscope.pipelines import pipeline
from librosa import load
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
    model_revision='v1.0.2',
    
)

def speaker_verification(wav1, wav2):
    new_sr = sv_pipeline.model_config['sample_rate']
    wav1, _ = load(wav1, sr=new_sr)
    wav2, _ = load(wav2, sr=new_sr)
    result = sv_pipeline([wav1, wav2])
    print(f"Verify {wav1} and {wav2}: {result}")
    return result['score']

if __name__ == '__main__':
    # speaker1_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_a_cn_16k.wav'
    # speaker1_b_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_b_cn_16k.wav'
    # speaker2_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker2_a_cn_16k.wav'
    
    
    speaker1_a_wav = r"..\data\antona\Tower01_Antuona_02.wav"
    speaker1_b_wav = r"..\data\antona\Tower01_Antuona_03.wav"
    speaker2_a_wav = r"..\data\antona\Tower01_Antuona_04.wav"
    # resample_wav to 16k
   
    # 相同说话人语音
    result = speaker_verification(speaker1_a_wav, speaker1_b_wav)
    print(result)
    # 不同说话人语音
    result = speaker_verification(speaker1_a_wav, speaker2_a_wav)
    print(result)
    # 可以自定义得分阈值来进行识别
    result = speaker_verification(speaker1_a_wav, speaker2_a_wav)
    print(result)