import os
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from openmodal.util.text.languages.symbols import language_tone_start_map
# from audio.symbols import punctuation

from openmodal.model.text.pretrained_bert.tone_sandhi import ToneSandhi
from openmodal.util.text.languages.english import g2p as g2p_en
from transformers import AutoTokenizer

from openmodal.util.text.languages.zh_normalization.text_normlization import TextNormalizer

punctuation = ["!", "?", "…", ",", ".", "'", "-"]
current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join("openmodal\\util\\text\\languages\\opencpop-strict.txt")).readlines()
}

import jieba.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "“": "'",
    "”": "'",
    "‘": "'",
    "’": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):
    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)
    replaced_text = re.sub(r"[^\u4e00-\u9fa5_a-zA-Z\s" + "".join(punctuation) + r"]+", "", replaced_text)
    replaced_text = re.sub(r"[\s]+", " ", replaced_text)

    return replaced_text


def g2p(text,ckpt_bert_path='bert-base-multilingual-uncased'):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if not re.match(r'^[\s\.,;:!?。，！？；：]*$', i)]
    phones, tones, word2ph = _g2p(sentences,ckpt_bert_path)
    assert sum(word2ph) == len(phones)
    # assert len(word2ph) == len(audio)  # Sometimes it will crash,you can add a try-catch.
    phones = ["_"] + phones + ["_"]
    tones = [0] + tones + [0]
    word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _get_initials_finals(word):
    initials = []
    finals = []
    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def text_normalize(text):
    # numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    # for number in numbers:
    #     text = text.replace(number, cn2an.an2cn(number), 1)
    # text = replace_punctuation(text)
    # return text
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    tx = TextNormalizer()
    sentences = tx.normalize(text)
    dest_text = ""
    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # 避免重复标点引起的参考泄露
    dest_text = replace_consecutive_punctuation(dest_text)
    return dest_text

def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in punctuation)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result

def get_bert_feature(text, word2ph,ckpt_bert_path, device):
    from openmodal.model.text.pretrained_bert import chinese_bert
    return chinese_bert.get_bert_feature(text, word2ph, ckpt_bert_path, device=device)

from openmodal.util.text.languages.chinese import _g2p as _chinese_g2p
def _g2p(segments,ckpt_bert_path='bert-base-multilingual-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_bert_path,clean_up_tokenization_spaces=True)
    spliter = '#$&^!@'

    phones_list = []
    tones_list = []
    word2ph = []

    for text in segments:
        assert spliter not in text
        # replace all english words
        text = re.sub('([a-zA-Z\s]+)', lambda x: f'{spliter}{x.group(1)}{spliter}', text)
        texts = text.split(spliter)
        texts = [t for t in texts if len(t) > 0]

        
        for text in texts:
            if re.match('[a-zA-Z\s]+', text):
                # english
                tokenized_en = tokenizer.tokenize(text)
                phones_en, tones_en, word2ph_en = g2p_en(text=None, pad_start_end=False, tokenized=tokenized_en)
                # apply offset to tones_en
                tones_en = [t + language_tone_start_map['EN'] for t in tones_en]
                phones_list += phones_en
                tones_list += tones_en
                word2ph += word2ph_en
            else:
                phones_zh, tones_zh, word2ph_zh = _chinese_g2p([text])
                phones_list += phones_zh
                tones_list += tones_zh
                word2ph += word2ph_zh
    return phones_list, tones_list, word2ph

    

if __name__ == "__main__":
    # from audio.chinese_bert import get_bert_feature

    text = "NFT啊！chemistry 但是《原神》是由,米哈\游自主，  [研发]的一款全.新开放世界.冒险游戏"
    text = '我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。'
    text = '今天下午，我们准备去shopping mall购物，然后晚上去看一场movie。'
    text = '我们现在 also 能够 help 很多公司 use some machine learning 的 algorithms 啊!'
    text = text_normalize(text)
    print(text)
    # phones, tones, word2ph = g2p(text, impl='v2')
    # bert = get_bert_feature(text, word2ph, device='cuda:0')
    # print(phones)
    # import pdb; pdb.set_trace()


# # 示例用法
# audio = "这是一个示例文本：,你好！这是一个测试...."
# print(g2p_paddle(audio))  # 输出: 这是一个示例文本你好这是一个测试
