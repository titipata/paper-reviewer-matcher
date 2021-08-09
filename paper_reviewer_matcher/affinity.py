import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from .vectorizer import LogEntropyVectorizer, BM25Vectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

__all__ = ["compute_topics",
           "affinity_computation",
           "create_lp_matrix",
           "create_assignment"]

def compute_topics(
    papers,
    weighting='tfidf',
    projection='svd',
    min_df=3, max_df=0.8,
    lowercase=True, norm='l2',
    analyzer='word', token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    n_components=30,
    stop_words='english'
):
    """
    Compute topics
    """
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

    X = model.fit_transform(papers) # weighting matrix

    # topic modeling
    if projection == 'svd':
        topic_model = TruncatedSVD(n_components=n_components, algorithm='arpack')
        X_topic = topic_model.fit_transform(X)
    elif projection == 'pca':
        topic_model = PCA(n_components=n_components)
        X_topic = topic_model.fit_transform(X.todense())
    else:
        print("select projection from ['svd', 'pca']")
    return X_topic


def compute_affinity(papers, reviewers,
                     weighting='tfidf',
                     projection='svd',
                     min_df=3, max_df=0.8,
                     distance='euclidean',
                     lowercase=True, norm='l2',
                     token_pattern=r'\w{1,}',
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
    distance: str, either 'euclidean' or 'cosine' distance


    Returns
    -------
    A: ndarray, affinity array from given papers and reviewers
    """
    n_papers = len(papers)

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

    if distance == 'euclidean':
        A = - euclidean_distances(paper_vectors, reviewer_vectors) # dense affinity matrix
    elif distance == 'cosine':
        A = - cosine_distances(paper_vectors, reviewer_vectors) # dense affinity matrix
    else:
        A = None
        print("Distance function can only be selected from `euclidean` or `cosine`")

    return A

def create_lp_matrix(A, min_reviewers_per_paper=0, max_reviewers_per_paper=10,
                        min_papers_per_reviewer=0, max_papers_per_reviewer=10):
    """
    The problem formulation of paper-reviewer matching problem is as follow:
    we want to maximize this cost function with constraint

        maximize A.T * b
        subject to N_p * b <= c_p (c_p = maximum number of reviewer per paper)
                   N_r * b <= c_r (c_r = maximum number of paper per reviewer)
                   b <= 1
                   b >= 0

    This problem can be reformulate as
        maximize A.T * b
        subject to K * b <= d
        where K = [N_p; N_r; I; -I] and d = [c_p, c_r, 1, 0]

    where A is an affinity matrix (e.g. topic distance matrix)
          N is node edge adjacency matrix, N = [N_p; N_r; I; -I]
          d is column constraint vector, d = [c_p, c_r, 1, 0]

    Reference
    ---------
    Taylor, Camillo J. "On the optimal assignment of conference papers to reviewers." (2008).
    """
    n_papers, n_reviewers = A.shape
    n_edges = np.count_nonzero(A)

    i, j = A.nonzero()
    v = A[i, j]

    N_e = sp.dok_matrix((n_papers + n_reviewers, n_edges), dtype=np.float)
    N_e[i, range(n_edges)] = 1
    N_e[j + n_papers, range(n_edges)] = 1

    N_p = sp.dok_matrix((n_papers, n_edges), dtype=np.int)
    N_p[i, range(n_edges)] = -1

    N_r = sp.dok_matrix((n_reviewers, n_edges), dtype=np.int)
    N_r[j, range(n_edges)] = -1

    K = sp.vstack([N_e, N_p, N_r, sp.identity(n_edges), -sp.identity(n_edges)])

    d = [max_reviewers_per_paper] * n_papers + [max_papers_per_reviewer] * n_reviewers + \
        [-min_reviewers_per_paper] * n_papers + [-min_papers_per_reviewer] * n_reviewers + \
        [1] * n_edges + [0] * n_edges
    d = np.atleast_2d(d).T # column constraint vector

    return v, K, d

def create_assignment(x_sol, A):
    """
    Given a solution from linear programming problem for paper assignments
    with affinity matrix A, produce the actual assignment matrix b
    """
    n_papers, n_reviewers = A.shape
    i, j = A.nonzero()
    t = np.array(x_sol > 0.5).flatten()
    b = np.zeros((n_papers, n_reviewers))
    b[i[t], j[t]] = 1
    return b
