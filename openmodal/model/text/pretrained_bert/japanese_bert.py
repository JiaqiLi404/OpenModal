import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys


models = {}
tokenizers = {}
def get_bert_feature(text, word2ph, ckpt_bert_path='tohoku-nlp/bert-base-japanese-v3',device=None):
    global model
    global tokenizer

    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if ckpt_bert_path not in models:
        model = AutoModelForMaskedLM.from_pretrained(ckpt_bert_path).to(
            device
        )
        models[ckpt_bert_path] = model
        tokenizer = AutoTokenizer.from_pretrained(ckpt_bert_path)
        tokenizers[ckpt_bert_path] = tokenizer
    else:
        model = models[ckpt_bert_path]
        tokenizer = tokenizers[ckpt_bert_path]


    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        tokenized = tokenizer.tokenize(text)
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    assert inputs["input_ids"].shape[-1] == len(word2ph), f"{inputs['input_ids'].shape[-1]}/{len(word2ph)}"
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
