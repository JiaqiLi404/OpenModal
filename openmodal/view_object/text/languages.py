from openmodal.view_object.BaseEnum import BaseEnum


class LanguagesEnum(BaseEnum):
    ZH = "ZH"
    ZH_CA = "ZH_CANTONESE"
    JP = "JP"
    EN = "EN"
    ZH_MIX_EN = "ZH_MIX_EN"
    KR = "KR"
    ES = "ES"
    SP = "SP"
    FR = "FR"
    RU = "RU"
    DE= "DE"

    @classmethod
    def from_str(cls,text):
        text=text.upper()
        if text == "ZH":
            return cls.ZH
        elif text == "ZH_CANTONESE":
            return cls.ZH_CA
        elif text == "JP" or text == "JA":
            return cls.JP
        elif text == "EN":
            return cls.EN
        elif text == "ZH_MIX_EN":
            return cls.ZH_MIX_EN
        elif text == "KR" or text == "KO":
            return cls.KR
        elif text == "ES":
            return cls.ES
        elif text == "SP":
            return cls.SP
        elif text == "FR":
            return cls.FR
        elif text == "RU":
            return cls.RU
        elif text == "DE":
            return cls.DE
        else:
            raise ValueError("Invalid language: {}".format(text))
