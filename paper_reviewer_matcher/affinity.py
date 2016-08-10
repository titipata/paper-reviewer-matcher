from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .vectorizer import LogEntropyVectorizer, BM25Vectorizer
from sklearn.decomposition import PCA, TruncatedSVD

__all__ = ["affinity_computation"]

def affinity_computation(papers, reviewers,
                         weighting='tfidf',
                         projection='svd',
                         min_df=3, max_df=0.8,
                         lowercase=True, norm='l2',
                         strip_accents='unicode',
                         analyzer='word', token_pattern=r'\w{1,}',
                         ngram_range=(1, 1),
                         n_components=30):
    """
    Create affinity matrix (distance matrix)
    from given list of papers' abstract and reviewers' abstract

    """

    if weighting == 'count':
        model = CountVectorizer(min_df=min_df, max_df=max_df,)
    elif weighting == 'tfidf':
        model = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                lowercase=lowercase, norm=norm,
                                strip_accents=strip_accents, analyzer=analyzer,
                                token_pattern=token_pattern, ngram_range=ngram_range,
                                use_idf=True, smooth_idf=True, sublinear_tf=True,
                                stop_words=stop_words)
    elif weighting == 'entropy':
        model =
    elif weighting == 'bm25':
        model =
    else:
        print('select weighting scheme from ')

    X = model.fit_transform(papers + reviewers) # weighting matrix

    # topic modeling
    topic_model = PCA(n_components=n_components)
    topic_model.fit_transform(X)

    # compute affinity matrix

    return A
