from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .vectorizer import LogEntropyVectorizer, BM25Vectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

__all__ = ["affinity_computation"]

def affinity_computation(papers, reviewers,
                         weighting='tfidf',
                         projection='svd',
                         min_df=3, max_df=0.8,
                         lowercase=True, norm='l2',
                         analyzer='word', token_pattern=r'\w{1,}',
                         ngram_range=(1, 1),
                         n_components=30,
                         stop_words='english'):
    """
    Create affinity matrix (or distance matrix)
    from given list of papers' abstract and reviewers' abstract

    Parameters
    ----------
    papers: list, list of string (incoming paper for the conference)
    reviewers: list, list of string from reviewers (e.g. paper that they prefer)
    weighting: str, weighting scheme for count vector matrix
        this can be ('count', 'tfidf', 'entropy', 'bm25')
    projection: str, either 'svd' or 'pca' for topic modeling


    Returns
    -------
    A: ndarray, affinity array from given papers and reviewers
    """
    n_papers = len(papers)
    n_reviwers = len(reviewers)

    if weighting == 'count':
        model = CountVectorizer(min_df=min_df, max_df=max_df,
                                token_pattern=token_pattern,
                                ngram_range=ngram_range,
                                stop_words=stop_words)
    elif weighting == 'tfidf':
        model = TfidfVectorizer(min_df=min_df, max_df=max_df,
                                lowercase=lowercase, norm=norm,
                                token_pattern=token_pattern,
                                ngram_range=ngram_range,
                                use_idf=True, smooth_idf=True, sublinear_tf=True,
                                stop_words=stop_words)
    elif weighting == 'entropy':
        model = LogEntropyVectorizer(min_df=min_df, max_df=max_df,
                                     lowercase=lowercase,
                                     token_pattern=token_pattern,
                                     ngram_range=ngram_range,
                                     stop_words=stop_words)
    elif weighting == 'bm25':
        model = BM25Vectorizer(min_df=min_df, max_df=max_df,
                               lowercase=lowercase,
                               token_pattern=token_pattern,
                               ngram_range=ngram_range,
                               stop_words=stop_words)
    else:
        print("select weighting scheme from ['count', 'tfidf', 'entropy', 'bm25']")

    X = model.fit_transform(papers + reviewers) # weighting matrix

    # topic modeling
    if projection == 'svd':
        topic_model = TruncatedSVD(n_components=n_components, algorithm='arpack')
        X_topic = topic_model.fit_transform(X)
    elif projection == 'pca':
        topic_model = PCA(n_components=n_components)
        X_topic = topic_model.fit_transform(X.todense())
    else:
        print("select projection from ['svd', 'pca']")

    # compute affinity matrix
    paper_vectors = X_topic[:n_papers, :]
    reviewer_vectors = X_topic[n_papers:, :]

    A = -euclidean_distances(paper_vectors, reviewer_vectors) # dense affinity matrix

    return A
