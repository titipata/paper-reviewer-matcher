"""
=========================
neuromatch Group Matching
=========================

Group matching script and its output. We read dataset from Cloud Firestore
and export as CSV, and JSON file.

TODO: make it as a script, add documentation on how it works
"""

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage
import hcluster   # requires dedupe-hcluster
from paper_reviewer_matcher import (
    preprocess, compute_affinity
)


def compute_conflicts(df):
    """
    Compute conflict for a given dataframe
    """
    cois = []
    for i, r in tqdm(df.iterrows()):
        exclude_list = r['conflicts'].split(';')
        for j, r_ in df.iterrows():
            if max([fuzz.ratio(r_['fullname'], n) for n in exclude_list]) >= 85:
                cois.append([i, j])
                cois.append([j, i])
    return cois

def generate_pod_numbers(n_users, n_per_group):
    """
    Generate pod numbers in sequence
    """
    groups = []
    for i in range(1, int(n_users / n_per_group) + 2):
        groups.extend([i] * n_per_group)
    groups = groups[:n_users]
    return groups


if __name__ == '__main__':
    users = pd.read_csv('data/mindmatch_example.csv').to_dict(orient='records')
    n_users = len(users)
    print('Number of registered users: {}'.format(n_users))

    users_df = pd.DataFrame(users).fillna('')
    users_dict = {r['user_id']: dict(r) for _, r in users_df.iterrows()}  # map of user id to details
    persons_1 = list(map(preprocess, list(users_df['abstracts'])))
    persons_2 = list(map(preprocess, list(users_df['abstracts'])))
    A = compute_affinity(
        persons_1, persons_2,
        n_components=30, min_df=2, max_df=0.8,
        weighting='tfidf', projection='svd'
    )
    cois_list = compute_conflicts(users_df)
    for i, j in cois_list:
        A[i, j] = -1

    A_cluster = - A
    A_cluster[A_cluster == 1000] = 1
    A_rand = np.random.randn(n_users, n_users) * 0.01 * A_cluster.var() # add randomness

    z = linkage(A_cluster + A_rand,
                method='average',
                metric='euclidean',
                optimal_ordering=True)
    cluster = hcluster.fcluster(z, t=0.01,
                                criterion='distance') # distance
    users_group_df['cluster'] = cluster
    users_sorted_df  = users_group_df.sort_values('cluster')
    cluster_numbers = generate_pod_numbers(n_users=len(users_sorted_df), n_per_group=5)
    users_sorted_df['cluster'] = cluster_numbers
    users_sorted_df.to_csv('group_matching_users.csv', index=False)
