import re
from pypinyin import lazy_pinyin, Style

def normalize_text(text):
    # Remove Chinese punctuation
    text = text.replace("嗯", "恩").replace("呣", "母")
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    # Remove any extra spaces (though these are rare in Chinese text)
    text = re.sub(r'\s+', '', text)
    return text
    

def text_to_phonemes(text):
    phonemes = lazy_pinyin(text, style=Style.TONE3)
    return phonemes

def normalized_text_to_phonemes(text):
    normalized_text = normalize_text(text)
    phonemes = text_to_phonemes(normalized_text)
    return phonemes

if __name__ == "__main__":

    texts = ["啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏", "呣呣呣～就是…大人的鼹鼠党吧？", "你好", 'I love you', '你侵入實驗室把我偷出來是嚴重的違規行為']
    for text in texts:
        print(normalize_text(text))
        print(normalized_text_to_phonemes(text))