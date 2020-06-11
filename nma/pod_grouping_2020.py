import re
from math import atan2
import numpy as np
import pandas as pd
import paper_reviewer_matcher as pp
from paper_reviewer_matcher import preprocess, affinity_computation, \
                                   create_lp_matrix, create_assignment
from scipy.cluster.hierarchy import linkage
import hcluster
from sklearn.preprocessing import MinMaxScaler
import copkmeans
from itertools import combinations, permutations, product
from tqdm import tqdm, tqdm_notebook

from sklearn.manifold import MDS
from copkmeans.cop_kmeans import cop_kmeans

selected_cols = [
    'index', 'gender', 'institution', 'home_country',
    'institute_city', 'residence_country',
    'timezone', 'second_timezone', 'third_timezone',
    'Statement'
]


def remove_text_parentheses(text):
    """
    Remove text inside parentheses
    """
    return re.sub(r"[\(\[].*?[\)\]]", "", text).strip()


def compute_tz_distance(node_1, node_2):
    """
    Compute timezone distance

    TODO: tweak distance between timezone
    """
    if node_1[0] == node_2[0] and node_1[1] == node_2[1]:
        return 0
    if node_1[0] == node_2[0] and node_1[1] != node_2[1]:
        return 5
    else:
        return 20


def calculate_timezone_distance(preferred_tz):
    """
    Sending array and distance function
    then calculate distance matrix as an output
    """
    D_preferred_tz = []
    for tz1 in preferred_tz:
         D_preferred_tz.append([compute_tz_distance(tz1, tz2) for tz2 in preferred_tz])
    D_preferred_tz = np.array(D_preferred_tz)
    return D_preferred_tz


def generate_pod_numbers(n_students=2157, n_per_group=18):
    """
    Generate pod numbers in sequence
    """
    groups = []
    for i in range(1, int(n_students / n_per_group) + 2):
        groups.extend([i] * n_per_group)
    groups = groups[:n_students]
    return groups


def calculate_geo_distance(lat1, lng1, lat2, lng2, R=6373.0):
    """
    Calculate geolocation in kilometers between two geolocation
    """
    d_lng = lng1 - lng2
    d_lat = lat1 - lat2
    a = np.sin(d_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lng / 2)**2
    c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance


def check_if_overlap(r1, r2,
                     cols_tz=['timezone', 'second_timezone', 'third_timezone']):
    """
    Check if slots have overlap
    if some of the slots overlap, return True,
    else return False
    """
    r1_ = [e for e in r1[cols_tz].fillna('').values
           if 'Slot' in e]
    r2_ = [e for e in r2[cols_tz].fillna('').values
           if 'Slot' in e]
    return any([tz1 == tz2 for tz1, tz2 in product(r1_, r2_)])


def generate_cannot_link_list(df, cols_tz=['timezone', 'second_timezone', 'third_timezone']):
    """
    Return list of cannot link tuple between indices e.g.
    [(1, 10), (10, 1), ...]
    """
    cannot_link = []
    for i, r1 in tqdm_notebook(df[cols].iterrows()):
        for j, r2 in df[cols].iterrows():
            if not check_if_overlap(r1, r2):
                cannot_link.append((i, j))
    return cannot_link


if __name__ == '__main__':
    # starter
    df = pd.read_csv('nma_applicants.csv', index=False)
    scaler = MinMaxScaler()


    # calculate timezone distance
    preferred_tz = df['timezone'].map(lambda t: remove_text_parentheses(t).split(' ')[-1])
    D_tz = calculate_timezone_distance(preferred_tz)

    # calculate geolocation distance
    lat_lng_df = df[['institute_longitude', 'institute_latitude']].rename(
        columns={'institute_longitude': 'lng', 'institute_latitude': 'lat'}
    )
    D_lat_lng = []
    for _, r1 in lat_lng_df.iterrows():
        D_lat_lng.append([
            calculate_geo_distance(r1.lat, r1.lng, r2.lat, r2.lng)
                                    for _, r2 in lat_lng_df.iterrows()
        ])
    D_lat_lng_scale = scaler.fit_transform(D_lat_lng)
    D_lat_lng_scale = pd.DataFrame(D_lat_lng_scale).fillna(np.nanmean(D_lat_lng_scale)).values

    # calculate topic distance between statement
    persons_1 = list(map(preprocess, list(df['Statement'])))
    persons_2 = list(map(preprocess, list(df['Statement'])))
    D_statement = - affinity_computation(persons_1, persons_2,
                                        n_components=30, min_df=2, max_df=0.8,
                                        weighting='tfidf', projection='svd')
    std_topic = D_statement.std()

    # clustering
    D_final = (D_statement) + (10 * std_topic * D_tz) + (std_topic * D_lat_lng_scale) # final distance
    X_mds = MDS(n_components=30).fit_transform(D_final)
    clusters_kmean, centers_kmean = cop_kmeans(dataset=X_mds, k=200, cl=cannot_link)
    output_df = df[selected_cols]
    output_df['pod_number'] = clusters_kmean

    # rearrange
    df_rearrange = []
    pod_num = 1
    for _, df_tz in output_df.groupby('timezone'):
        for _, df_pod_num in df_tz.groupby('pod_number'):
            df_pod_num['pod_number'] = pod_num
            df_rearrange.append(df_pod_num)
            pod_num += 1
    df_rearrange = pd.concat(df_rearrange)[selected_cols]
    df_rearrange.to_csv('pod_matching_rearrange_mds.csv', index=False)