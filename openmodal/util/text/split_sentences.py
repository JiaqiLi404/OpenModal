import re

class SplitSentenceStrategyEnum:
    SplitByLength = 'SplitByLength'
    SplitByPunctuation = 'SplitByPunctuation'
    SplitBySentence = 'SplitBySentence'

    @staticmethod
    def build(key:str):
        if key==SplitSentenceStrategyEnum.SplitByLength:
            return SplitSentenceStrategyEnum.SplitByLength
        elif key==SplitSentenceStrategyEnum.SplitByPunctuation:
            return SplitSentenceStrategyEnum.SplitByPunctuation
        elif key==SplitSentenceStrategyEnum.SplitBySentence:
            return SplitSentenceStrategyEnum.SplitBySentence
        else:
            raise ValueError(f"Invalid key {key}")


def split_sentence(
        text,
        strategy: SplitSentenceStrategyEnum = SplitSentenceStrategyEnum.SplitByLength,
        language_str='EN',
        min_len=10,
        max_len=512,
        by_sentence_length=4,
        by_length_desired=256
):
    # pre-process text, replace some punctuations
    # text = re.sub('[。！？；]', '.', text)
    # text = re.sub('[，]', ',', text)
    # text = re.sub('[：]', ':', text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub(r'\n\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub(r"……", ".", text)
    text = re.sub(r"——", ",", text)
    text = re.sub(r'[""]', '"', text)
    text = re.sub(r'([,.?!])', r'\1 ', text)
    text = text.strip("\n")
    splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }
    if text[-1] not in splits:
        text += "." if language_str in ['EN', 'FR', 'ES', 'SP'] else "。"

    if strategy == SplitSentenceStrategyEnum.SplitByLength:
        sentences = split_sentences_by_length(
            text,
            min_len=min_len,
            max_len=max_len,
            by_length_desired=by_length_desired,
            language=language_str
        )
        return sentences
    elif strategy == SplitSentenceStrategyEnum.SplitBySentence:
        # 在标点符号后添加一个空格
        text = re.sub('([,.!?;，。！？；])', r'\1 $#!', text)
        # 分隔句子并去除前后空格
        sentences = [s.strip() for s in text.split('$#!')]
        if len(sentences[-1]) == 0: del sentences[-1]
        sentences = [s for s in sentences if len(s) > 0 and not re.match(r'^[\s\.,;:!?。，！？；：]*$', s)]

        i = 0
        res = []
        while i < len(sentences):
            by_sentence_len = by_sentence_length
            res_temp = "".join(sentences[i:i + by_sentence_len]).strip()
            while by_sentence_len + i + 1 < len(sentences) and len(res_temp) < min_len:
                by_sentence_len += 1
                res_temp = "".join(sentences[i:i + by_sentence_len]).strip()
            while by_sentence_len >= 1 and len(res_temp) > max_len:
                by_sentence_len -= 1
                res_temp = "".join(sentences[i:i + by_sentence_len]).strip()
            res.append(res_temp)
            i += by_sentence_len
        return res
    elif strategy == SplitSentenceStrategyEnum.SplitByPunctuation:
        sentences = split_sentences_by_punctuations(
            text,
            min_len=min_len,
            max_len=max_len,
            language=language_str
        )
        return sentences
    else:
        raise ValueError("Invalid split strategy")


def split_sentences_by_length(text, min_len, max_len, by_length_desired, language):
    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    '''
    使用 seek(1) 获取下一个字符，并更新 current。
    如果当前片段长度 current 达到 max_len，并且切分点 split_pos 不为空且片段长度大于 by_length_desired 的一半，则向回找最近的切分点；否则，向回查找直到遇到合适的切分点。
    如果不是在引号中，并且当前字符是句号、问号、换行符或逗号，并且下一个字符是换行符或空格，则可能需要切分。
    如果是在引号中，并且下一个字符是引号且接下来的字符是换行符或空格，则跳过引号并进行切分。
    '''
    while pos < end_pos:
        c = seek(1)
        if len(current) >= max_len:
            if len(split_pos) > 0 and len(current) > (by_length_desired / 2):
                d = pos - split_pos[-1]
                seek(-d)
            else:
                while c not in '!?.\n ！？。' and pos > 0 and len(current) > by_length_desired:
                    c = seek(-1)
            commit()
        elif not in_quote and (c in '!?\n！？。' or (c in '.,。，' and peek(1) in '\n ')):
            while pos < len(text) - 1 and len(current) < max_len and peek(1) in '!?.！？。':
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= by_length_desired:
                commit()
        elif in_quote and peek(1) == '"“”‘’' and peek(2) in '\n ':
            seek(2)
            split_pos.append(pos)
    rv.append(current)
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r'^[\s\.,;:!?。，！？；：]*$', s)]
    rv = [item.strip() for item in rv if item.strip()]

    return rv


def split_sentences_by_punctuations(text, min_len, max_len, language):
    # 在标点符号后添加一个空格
    text = re.sub('([,.!?;，。！？；])', r'\1 $#!', text)
    # 分隔句子并去除前后空格
    # sentences = [s.strip() for s in re.split('(。|！|？|；)', audio)]
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        # todo: Sentence length could still be longer than max_len
        # if count_len>max_len:
        if count_len > min_len or ind == len(sentences) - 1 or (
                ind + 1 < len(sentences) and len(sentences[ind + 1]) > max_len):
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)


def merge_short_sentences_zh(sens):
    # return sens
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out


if __name__ == '__main__':
    zh_text = "好的，我来给你讲一个故事吧。从前有一个小姑娘，她叫做小红。小红非常喜欢在森林里玩耍，她经常会和她的小伙伴们一起去探险。有一天，小红和她的小伙伴们走到了森林深处，突然遇到了一只凶猛的野兽。小红的小伙伴们都吓得不敢动弹，但是小红并没有被吓倒，她勇敢地走向野兽，用她的智慧和勇气成功地制服了野兽，保护了她的小伙伴们。从那以后，小红变得更加勇敢和自信，成为了她小伙伴们心中的英雄。"
    en_text = "I didn’t know what to do. I said please kill her because it would be better than being kidnapped,” Ben, whose surname CNN is not using for security concerns, said on Wednesday. “It’s a nightmare. I said ‘please kill her, don’t take her there.’"
    sp_text = "¡Claro! ¿En qué tema te gustaría que te hable en español? Puedo proporcionarte información o conversar contigo sobre una amplia variedad de temas, desde cultura y comida hasta viajes y tecnología. ¿Tienes alguna preferencia en particular?"
    fr_text = "Bien sûr ! En quelle matière voudriez-vous que je vous parle en français ? Je peux vous fournir des informations ou discuter avec vous sur une grande variété de sujets, que ce soit la culture, la nourriture, les voyages ou la technologie. Avez-vous une préférence particulière ?"

    print(split_sentence(zh_text, language_str='ZH'))
    print(split_sentence(en_text, language_str='EN'))
    print(split_sentence(sp_text, language_str='SP'))
    print(split_sentence(fr_text, language_str='FR'))
