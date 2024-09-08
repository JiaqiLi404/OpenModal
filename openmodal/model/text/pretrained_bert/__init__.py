from openmodal.view_object.text.languages import LanguagesEnum


def get_bert(norm_text, word2ph, language, ckpt_bert_path, device):
    from openmodal.model.text.pretrained_bert.chinese_bert import get_bert_feature as zh_bert
    from openmodal.model.text.pretrained_bert.english_bert import get_bert_feature as en_bert
    from openmodal.model.text.pretrained_bert.japanese_bert import get_bert_feature as jp_bert
    from openmodal.util.text.languages.chinese_mix import get_bert_feature as zh_mix_en_bert
    from openmodal.model.text.pretrained_bert.spanish_bert import get_bert_feature as sp_bert
    from openmodal.model.text.pretrained_bert.french_bert import get_bert_feature as fr_bert
    from openmodal.util.text.languages.korean import get_bert_feature as kr_bert

    lang_bert_func_map = {LanguagesEnum.ZH: zh_bert, LanguagesEnum.EN: en_bert, LanguagesEnum.JP: jp_bert,
                          LanguagesEnum.ZH_MIX_EN: zh_mix_en_bert, LanguagesEnum.FR: fr_bert, LanguagesEnum.SP: sp_bert,
                          LanguagesEnum.ES: sp_bert, LanguagesEnum.KR: kr_bert,LanguagesEnum.ZH_CA: zh_bert}
    bert = lang_bert_func_map[language](norm_text, word2ph, ckpt_bert_path, device)
    return bert
