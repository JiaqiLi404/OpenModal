from openmodal.model.text.pretrained_bert import chinese_mix
from openmodal.util.text.languages import french, spanish, english, japanese, chinese, korean
from openmodal.util.text.languages.symbols import num_zh_tones, num_ja_tones, num_en_tones, num_kr_tones, num_es_tones, \
    zh_symbols, ja_symbols, en_symbols, kr_symbols, es_symbols, fr_symbols, de_symbols, ru_symbols, symbols, \
    language_tone_start_map

language_module_map = {"ZH": chinese, "JP": japanese, "EN": english, 'ZH_MIX_EN': chinese_mix, 'KR': korean,
                    'FR': french, 'SP': spanish, 'ES': spanish}


def clean_text(text, language):
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones, tones, word2ph = language_module.g2p(norm_text)
    return norm_text, phones, tones, word2ph


_symbol_to_id = {s: i for i, s in enumerate(symbols)}

# language maps
language_id_map = {"ZH": 0, "JP": 1, "EN": 2, "ZH_MIX_EN": 3, 'KR': 4, 'ES': 5, 'SP': 5 ,'FR': 6}


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