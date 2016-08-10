# weighting function
# from https://github.com/titipata/science_concierge/blob/master/science_concierge/vectorizer.py

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted

class LogEntropyVectorizer(CountVectorizer):
    """Log-entropy vectorizer
    Convert collection of raw documents to matrix of log-entropy features
    Adds on functionality for scikit-learn CountVectorizer to
    calculate log-entropy term matrix
    Log-entropy
    -----------
    Assume we have term i in document j can be calculated as follows
    Global entropy
        p_ij = f_ij / sum_j(f_ij)
        g_i = 1 + sum_j (p_ij * log p_ij / log n)
    log-entropy of term i in document j is
        l_ij = log(1 + f_ij) * g_i
    where
        f_ij is number of term i that appears in document j
        sum_j(f_ij) is total number of times term i occurs in
            the whole documents
        n is total number of documents
        g_i is sum of entropy across all documents j
    Parameters
    ----------
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words : string {'english'}, list, or None (default)
    lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    max_df : float in range [0, 1] or int, default=1.0
    min_df : float in range [0, 1] or int, default=1
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.
    smooth_idf: boolean, default=False
    See also
    --------
    CountVectorizer
        Tokenize the documents and count the occurrences of token and return
        them as a sparse matrix
    TfidfTransformer
        Apply Term Frequency Inverse Document Frequency normalization to a
        sparse matrix of occurrence counts.
    Example
    -------
    >> model = LogEntropyVectorizer(norm=None, ngram_range=(1,1))
    >> docs = ['this this this book',
               'this cat good',
               'cat good shit']
    >> X = model.fit_transform(docs)
    References
    ----------
        - https://en.wikipedia.org/wiki/Latent_semantic_indexing
        - http://webpages.ursinus.edu/akontostathis/KontostathisHICSSFinal.pdf
    """
    def __init__(self, encoding='utf-8', decode_error='strict',
                 lowercase=True, preprocessor=None, tokenizer=None,
                 analyzer='word', stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 vocabulary=None, binary=False,
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, norm='l2', smooth_idf=False):


        super(LogEntropyVectorizer, self).__init__(
            encoding=encoding,
            decode_error=decode_error,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
        )

        self.norm = norm
        self.smooth_idf = smooth_idf


    def fit(self, raw_documents, y=None):
        """Learn vocabulary and log-entropy from training set.
        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects
        Returns
        -------
        self : LogEntropyVectorizer
        """
        X = super(LogEntropyVectorizer, self).fit_transform(raw_documents)

        n_samples, n_features = X.shape
        gf = np.ravel(X.sum(axis=0)) # count total number of each words

        if self.smooth_idf:
            n_samples += int(self.smooth_idf)
            gf += int(self.smooth_idf)

        P = (X * sp.spdiags(1./gf, diags=0, m=n_features, n=n_features)) # probability of word occurence
        p = P.data
        P.data = (p * np.log2(p) / np.log2(n_samples))
        g = 1 + np.ravel(P.sum(axis=0))
        f = np.log2(1 + X.data)
        X.data = f
        # global weights
        self._G = sp.spdiags(g, diags=0, m=n_features, n=n_features)
        return self


    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)
        return self.transform(raw_documents)


    def transform(self, raw_documents):
        X = super(LogEntropyVectorizer, self).transform(raw_documents)
        check_is_fitted(self, '_G', 'global weight vector is not fitted')
        L = X * self._G  # sparse entropy matrix

        if self.norm is not None:
            L = normalize(L, norm=self.norm, copy=False)
        return L


class BM25Vectorizer(CountVectorizer):
    """
    Implementation of Okapi BM25
    Parameters
    ----------
    encoding : string, 'utf-8' by default.
        If bytes or files are given to analyze, this encoding is used to
        decode.
    decode_error : {'strict', 'ignore', 'replace'}
        Instruction on what to do if a byte sequence is given to analyze that
        contains characters not of the given `encoding`. By default, it is
        'strict', meaning that a UnicodeDecodeError will be raised. Other
        values are 'ignore' and 'replace'.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.
    stop_words : string {'english'}, list, or None (default)
    lowercase : boolean, default True
        Convert all characters to lowercase before tokenizing.
    token_pattern : string
        Regular expression denoting what constitutes a "token", only used
        if ``analyzer == 'word'``. The default regexp selects tokens of 2
        or more alphanumeric characters (punctuation is completely ignored
        and always treated as a token separator).
    max_df : float in range [0, 1] or int, default=1.0
    min_df : float in range [0, 1] or int, default=1
    b : float, default 0.75
        parameter for Okapi BM25
    k1 : float, suggested value from [1.2, 2.0]
        parameter for Okapi BM25
    References
    ----------
        - Okapi BM25 https://en.wikipedia.org/wiki/Okapi_BM25
        - Introduction to Information Retrieval http://nlp.stanford.edu/IR-book/essir2011/pdf/11prob.pdf
    """
    def __init__(self, encoding='utf-8', decode_error='strict',
                 lowercase=True, preprocessor=None, tokenizer=None,
                 analyzer='word', stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 vocabulary=None, binary=False,
                 ngram_range=(1, 1), max_df=1.0, min_df=1,
                 max_features=None, b=0.75, k1=1.5):

        super(BM25Vectorizer, self).__init__(
            encoding=encoding,
            decode_error=decode_error,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            analyzer=analyzer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
        )

        self.b = b
        self.k1 = k1

    def fit_transform(self, raw_documents, y=None):

        X = super(BM25Vectorizer, self).fit_transform(raw_documents)
        X = X.tocoo()
        n_samples, n_features = X.shape
        doc_len = np.ravel(X.sum(axis=1))
        avg_len = doc_len.mean()
        len_norm = 1.0 - self.b + (self.b * doc_len / avg_len)
        idf = np.log(float(n_samples) / (1 + np.bincount(X.col)))
        X.data = X.data * (self.k1 + 1.0) / (self.k1 * len_norm[X.row] + X.data) * idf[X.col]
        return X.tocsr()
