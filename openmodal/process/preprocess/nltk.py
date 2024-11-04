import nltk
from typing import List

def check_nltk(libs:List):
    for lib in libs:
        nltk.download(lib)