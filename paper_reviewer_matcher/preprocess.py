import re
import string
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer

__all__ = ["preprocess"]

stemmer = PorterStemmer()
w_tokenizer = WhitespaceTokenizer()
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))

def preprocess(text, stemming=True):
    """
    Apply Snowball stemmer to string
    Parameters
    ----------
    text : str, input abstract of papers/posters string
    stemming : boolean, apply Porter stemmer if True,
        default True
    """
    if isinstance(text, (type(None), float)):
        text_preprocess = ''
    else:
        text = unidecode(text).lower()
        text = punct_re.sub(' ', text) # remove punctuation
        if stemming:
            text_preprocess = [stemmer.stem(token) for token in w_tokenizer.tokenize(text)]
        else:
            text_preprocess = w_tokenizer.tokenize(text)
        text_preprocess = ' '.join(text_preprocess)
    return text_preprocess
