from openmodal.util.text.languages import french, spanish, english, japanese, chinese, korean, chinese_mix
from openmodal.util.text.languages.symbols import symbols, language_tone_start_map, language_id_map
from openmodal.view_object.text.languages import LanguagesEnum

language_module_map = {LanguagesEnum.ZH: chinese, LanguagesEnum.JP: japanese, LanguagesEnum.EN: english,
                       LanguagesEnum.ZH_MIX_EN: chinese_mix, LanguagesEnum.KR: korean,
                       LanguagesEnum.FR: french, LanguagesEnum.SP: spanish, LanguagesEnum.ES: spanish}


def clean_text(text, language, ckpt_bert_dir):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text) if hasattr(language_module,"text_normalize") else text
    phones, tones, word2ph = language_module.g2p(norm_text, ckpt_bert_dir) \
        if language == LanguagesEnum.ZH_MIX_EN or language == LanguagesEnum.ZH \
        else language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


_symbol_to_id = {s: i for i, s in enumerate(symbols)}


def cleaned_text_to_sequence(cleaned_text, tones, language, symbol_to_id=None):
    """Converts a string of audio to a sequence of IDs corresponding to the symbols in the audio.
    Args:
      audio: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the audio
    """
    symbol_to_id_map = symbol_to_id if symbol_to_id else _symbol_to_id
    phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
    tone_start = language_tone_start_map[language]
    tones = [i + tone_start for i in tones]
    lang_id = language_id_map[language]
    lang_ids = [lang_id for i in phones]
    return phones, tones, lang_ids
