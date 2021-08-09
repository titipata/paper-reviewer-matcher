import re
from math import atan2
import numpy as np
import pandas as pd
import paper_reviewer_matcher as pp
from paper_reviewer_matcher import (
    preprocess, compute_affinity,
    create_lp_matrix, create_assignment
)
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import MinMaxScaler

from itertools import product
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


def compute_tz_distance_dict(d1, d2):
    """
    Compute timezone distance
    """
    idx1 = d1['idx']
    idx2 = d2['idx']
    if d1['timezone'] == d2['timezone'] and d1['second_timezone'] == d2['second_timezone']:
        return (idx1, idx2, 0.0)
    elif d1['timezone'] == d2['timezone'] and d1['second_timezone'] != d2['second_timezone']:
        return (idx1, idx2, 0.3)
    elif d1['timezone'] == d2['timezone'] or d1['second_timezone'] == d2['second_timezone']\
        or d1['second_timezone'] == d2['timezone'] or d1['timezone'] == d2['second_timezone']:
        return (idx1, idx2, 0.3)
    else:
        return (idx1, idx2, 1.0)


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


def calculate_geo_distance(d1, d2, R=6373.0):
    """
    Calculate geolocation in kilometers between two geolocation
    """
    lat1, lng1 = d1['lat'], d1['lng']
    lat2, lng2 = d2['lat'], d2['lng']
    try:
        d_lng = lng1 - lng2
        d_lat = lat1 - lat2
        a = np.sin(d_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(d_lng / 2)**2
        c = 2 * atan2(np.sqrt(a), np.sqrt(1 - a))
        distance = R * c
        return (d1['idx'], d2['idx'], distance)
    except:
        return (d1['idx'], d2['idx'], np.nan)


def calculate_geo_distance_matrix(df):
    """
    Calculate geo distance matrix from a given dataframe
    """
    n_users = len(df)
    lat_lng_df = df[['idx', 'index', 'institute_longitude', 'institute_latitude']].rename(
        columns={'institute_longitude': 'lng', 'institute_latitude': 'lat'}
    )
    lat_lng_list = lat_lng_df.to_dict(orient='records')
    distance_df = pd.DataFrame(list(product(lat_lng_list, lat_lng_list)), columns=['loc1', 'loc2']).apply(
        lambda r: calculate_geo_distance(r['loc1'], r['loc2']), axis=1
    )
    d_fill = np.nanmean([d for _, _, d in distance_df.values])
    D_lat_lng = np.zeros((n_users, n_users))
    for idx1, idx2, d in distance_df.values:
        if not pd.isnull(d):
            D_lat_lng[idx1, idx2] = d
        else:
            D_lat_lng[idx1, idx2] = d_fill
    return D_lat_lng


def calculate_language_distance_matrix(df):
    """
    Calculate langugage distance matrix from a given dataframe

    The distance will be -0.5 if they have the same language preference
    """
    n_users = len(df)
    language_list = df[['idx', 'language']].to_dict(orient='records')
    D_language = np.zeros((n_users, n_users))
    for d1, d2 in product(language_list, language_list):
        if (d1['language'] or '') == (d2['language'] or '') and d1['idx'] != d2['idx']:
            D_language[d1['idx'], d2['idx']] = -0.5
    return D_language


def calculate_timezone_distance_matrix(df):
    """
    Calculate timezone distance matrix from a given dataframe
    """
    n_users = len(df)
    timezone_df = df[['idx', 'timezone', 'second_timezone']]
    timezone_df.loc[:, 'timezone'] = timezone_df.timezone.map(
        lambda t: remove_text_parentheses(t).split(' ')[-1]
    )
    timezone_df.loc[:, 'second_timezone'] = timezone_df.second_timezone.map(
        lambda t: remove_text_parentheses(t).split(' ')[-1].replace('me', ' ')
    )
    timezone_list = timezone_df.to_dict(orient='records')
    D_tz = np.zeros((n_users, n_users))
    for d1, d2 in product(timezone_list, timezone_list):
        idx1, idx2, tz_dist = compute_tz_distance_dict(d1, d2)
        D_tz[idx1, idx2] = tz_dist
    return D_tz


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


def check_if_timezone_overlap(d1, d2):
    """
    Check if two dictionary have overlap in timezone,
    if not, return an index between two dictionaries
    """
    tz_avail_1 = set([v for k, v in d1.items()
                      if (k != 'idx' and not pd.isnull(v) and v != '')])
    tz_avail_2 = set([v for k, v in d2.items()
                      if (k != 'idx' and not pd.isnull(v) and v != '')])
    if len(tz_avail_1.intersection(tz_avail_2)) == 0:
        return (d1['idx'], d2['idx'])
    else:
        return None


def generate_cannot_link_list(df, cols_tz=['timezone', 'second_timezone', 'third_timezone']):
    """
    Return list of cannot link tuple between indices e.g.
    [(1, 10), (10, 1), ...]
    """
    cols = ['index', 'timezone', 'second_timezone', 'third_timezone']
    cannot_link = []
    for i, r1 in tqdm_notebook(df[cols].iterrows()):
        for j, r2 in df[cols].iterrows():
            if not check_if_overlap(r1, r2):
                cannot_link.append((i, j))
    return cannot_link


def generate_cannot_list_list(df):
    """
    A more efficient way to generate cannot link list
    """
    cols_tz = ['idx', 'timezone', 'second_timezone', 'third_timezone']
    tz_df = df[cols_tz]
    tz_df.fillna('', inplace=True)
    tz_df['timezone'] = tz_df.timezone.map(lambda t: remove_text_parentheses(t).split(' ')[-1].replace('me', ' '))
    tz_df['second_timezone'] = tz_df.second_timezone.map(lambda t: remove_text_parentheses(t).split(' ')[-1].replace('me', ' '))
    tz_df['third_timezone'] = tz_df.third_timezone.map(lambda t: remove_text_parentheses(t).split(' ')[-1].replace('me', ' '))
    tz_list = tz_df.to_dict(orient='records')
    tz_pair_df = pd.DataFrame(list(product(tz_list, tz_list)), columns=['tz1', 'tz2'])
    cannot_link = list(tz_pair_df.apply(lambda r: check_if_timezone_overlap(r['tz1'], r['tz2']), axis=1).dropna())
    return cannot_link


if __name__ == '__main__':
    # starter
    scaler = MinMaxScaler()
    df = pd.read_csv('nma_applicants.csv', index=False)

    # calculate timezone distance
    D_tz = calculate_timezone_distance_matrix(df)

    # calculate geolocation distance
    D_lat_lng = calculate_geo_distance_matrix(df)
    D_lat_lng_scale = scaler.fit_transform(D_lat_lng)
    D_lat_lng_scale = pd.DataFrame(D_lat_lng_scale).fillna(np.nanmean(D_lat_lng_scale)).values

    # calculate topic distance between statement
    persons_1 = list(map(preprocess, list(df['Statement'])))
    persons_2 = list(map(preprocess, list(df['Statement'])))
    D_statement = - compute_affinity(persons_1, persons_2,
                                     n_components=30, min_df=2, max_df=0.8,
                                     weighting='tfidf', projection='svd')
    std_topic = D_statement.std()

    # list of cannot link
    cannot_link = generate_cannot_list_list(df)

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