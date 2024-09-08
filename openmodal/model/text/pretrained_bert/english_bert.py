import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import sys

tokenizer = None
model = None

def get_bert_feature(text, word2ph,ckpt_bert_path='bert-base-uncased', device=None):
    global model, tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_bert_path,clean_up_tokenization_spaces=True)
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if model is None:
        model = AutoModelForMaskedLM.from_pretrained(ckpt_bert_path).to(
            device
        )
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
