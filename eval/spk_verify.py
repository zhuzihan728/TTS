from modelscope.pipelines import pipeline
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
    model_revision='v1.0.2'
)

def speaker_verification(wav1, wav2):
    result = sv_pipeline([wav1, wav2])
    print(f"Verify {wav1} and {wav2}: {result}")
    return result['score']

if __name__ == '__main__':
    speaker1_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_a_cn_16k.wav'
    speaker1_b_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker1_b_cn_16k.wav'
    speaker2_a_wav = 'https://modelscope.cn/api/v1/models/damo/speech_campplus_sv_zh-cn_16k-common/repo?Revision=master&FilePath=examples/speaker2_a_cn_16k.wav'
    # 相同说话人语音
    result = speaker_verification(speaker1_a_wav, speaker1_b_wav)
    print(result)
    # 不同说话人语音
    result = speaker_verification(speaker1_a_wav, speaker2_a_wav)
    print(result)
    # 可以自定义得分阈值来进行识别
    result = speaker_verification(speaker1_a_wav, speaker2_a_wav)
    print(result)