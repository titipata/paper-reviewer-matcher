"""
Code snippet for producing CCN Mind Matching session 2019. 
We create affinity matrix of people-people using topic modeling 
then solve linear programming problem and apply networkx to solve the schedule problem

the given data includes the following columns
- RegistrantID
- NameFirst, first name of the attendee
- NameLast, last name of the attendee
- Affiliation
- Email
- mindMatchPersons, list of people attendee wants to meet (not used)
- RepresentativeWork
- mindMatchExclude
"""

import itertools
import numpy as np
import pandas as pd
import random
import networkx as nx
from itertools import chain
from collections import Counter
from fuzzywuzzy import fuzz
from scipy.sparse import coo_matrix
from ortools.linear_solver import pywraplp
from paper_reviewer_matcher import preprocess, affinity_computation, \
                                   create_lp_matrix, create_assignment
from docx import Document


def linprog(f, A, b):
    '''
    Solve the following linear programming problem
            maximize_x (f.T).dot(x)
            subject to A.dot(x) <= b
    where   A is a sparse matrix (coo_matrix)
            f is column vector of cost function associated with variable
            b is column vector
    '''

    # flatten the variable
    f = f.ravel()
    b = b.ravel()

    solver = pywraplp.Solver('SolveReviewerAssignment',
                             pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    infinity = solver.Infinity()
    n, m = A.shape
    x = [[]] * m
    c = [0] * n

    for j in range(m):
        x[j] = solver.NumVar(-infinity, infinity, 'x_%u' % j)

    # state objective function
    objective = solver.Objective()
    for j in range(m):
        objective.SetCoefficient(x[j], f[j])
    objective.SetMaximization()

    # state the constraints
    for i in range(n):
        c[i] = solver.Constraint(-infinity, int(b[i]))
        for j in A.col[A.row == i]:
            c[i].SetCoefficient(x[j], A.data[np.logical_and(A.row == i, A.col == j)][0])

    result_status = solver.Solve()
    if result_status != 0:
        print("The final solution might not converged")

    x_sol = np.array([x_tmp.SolutionValue() for x_tmp in x])

    return {'x': x_sol, 'status': result_status}


def build_line_graph(people):
    """
    Edge coloring and Vizing's theorem solution 
    can be found from Stack Overflow question below
    ref: https://stackoverflow.com/questions/51758406/creating-time-schedule-from-list-of-people-and-who-they-have-to-meet
    """
    G = nx.Graph()
    G.add_edges_from(((p, q) for p, L in people for q in L))
    return nx.line_graph(G)


def color_graph(G):
    return nx.greedy_color(G)


def format_answer(coloring):
    res = {}
    N = max(coloring.values()) + 1
    for meeting in coloring:
        time_slot = coloring[meeting]
        for meeting_member in (0, 1):
            if meeting[meeting_member] not in res:
                res[meeting[meeting_member]] = [None] * N
            res[meeting[meeting_member]][time_slot] = meeting[1-meeting_member]
    return res


def nest_answer(people, formatted):
    return [[p, formatted[p]] for p, v in people]


def split_exclude_string(people):
    """
    Function to split a given text of persons' name who wants to exclude 
    with comma separated for each name e.g. ``Konrad, Titipat``
    """
    people = people.replace('Mentor: ', '').replace('Lab-mates: ', '').replace('\r\n', ',').replace(';', ',')
    people_list = people.split(',')
    return [p.strip() for p in people_list if p.strip() is not '']


def create_coi_dataframe(df, people_maps, threshold=85, coreffered=True):
    """
    For a given dataframe of for mind-match people with 
    ``full_name``, ``mindMatchExcludeList`` column, and 
    a dictionary that map ``full_name`` to person_id, 
    create conflict of interest dataframe
    """
    coi_list = []
    for i, r in df.iterrows():
        if len(r['mindMatchExcludeList']) > 0:
            exclude_list = []
            for exclude in r['mindMatchExcludeList']:
                exclude_list.extend([
                    p['person_id'] for p in people_maps if 
                    exclude in p['full_name'] or 
                    fuzz.ratio(p['full_name'], exclude) >= threshold or 
                    fuzz.ratio(p['affiliation'], exclude) >= threshold]
                )
            exclude_list = sorted(pd.unique(exclude_list))
            if len(exclude_list) > 0:
                for e in exclude_list:
                    coi_list.append([i, e])

    # add extra co-referred COI for people who refers the same person
    coi_df = pd.DataFrame(coi_list, columns=['person_id', 'person_id_exclude'])

    if coreffered:
        coi_coreferred = [[g, list(g_df.person_id)] for g, g_df in coi_df.groupby(['person_id_exclude']) 
                        if len(list(g_df.person_id)) >= 2]

        coi_coreferred_list = []
        for _, exclude_list in coi_coreferred:
            coi_coreferred_list.extend(list(itertools.combinations(exclude_list, 2)))
        coi_coreferred_df = pd.DataFrame(coi_coreferred_list, columns=['person_id', 'person_id_exclude'])
        coi_df = pd.concat((coi_df, coi_coreferred_df))
        return coi_df
    else:
        return coi_df


if __name__ == '__main__':
    df = pd.read_csv('CN19_MindMatchData_20190903-A.csv', encoding='iso-8859-1')
    df['full_name'] = df['NameFirst'] + ' ' + df['NameLast']
    df['mindMatchExcludeList'] = df.mindMatchExclude.fillna(',').map(split_exclude_string)
    df['person_id'] = list(range(len(df)))

    people_maps = [{'person_id': r['person_id'], 
                    'full_name': r['full_name'], 
                    'affiliation': r['Affiliation']} 
                    for i, r in df.iterrows()]
    person_id_map = {r['person_id']: r['full_name'] for _, r in df.iterrows()}
    person_affil_map = {r['person_id']: r['Affiliation'] for _, r in df.iterrows()}
    registration_id_map = {r['person_id']: r['RegistrantID'] for _, r in df.iterrows()}
    coi_df = create_coi_dataframe(df, people_maps, threshold=85)

    # create assignment matrix
    n_meeting = 6
    persons_1 = list(map(preprocess, list(df['RepresentativeWork'])))
    persons_2 = list(map(preprocess, list(df['RepresentativeWork'])))
    A = affinity_computation(persons_1, persons_2,
                             n_components=10, min_df=2, max_df=0.8,
                             weighting='tfidf', projection='pca')
    # add constraints: conflict of interest
    A[np.arange(len(A)), np.arange(len(A))] = -1000
    for _, r in coi_df.iterrows():
        A[r['person_id'], r['person_id_exclude']] = -1000
        A[r['person_id_exclude'], r['person_id']] = -1000

    # trimming affinity matrix to reduce problem size
    A_trim = []
    for r in range(len(A)):
        a = A[r, :]
        a[np.argsort(a)[0:2]] = 0
        A_trim.append(a)
    A_trim = np.vstack(A_trim)

    v, K, d = create_lp_matrix(A_trim, 
                           min_reviewers_per_paper=6, max_reviewers_per_paper=6,
                           min_papers_per_reviewer=6, max_papers_per_reviewer=6)
    x_sol = linprog(v, K, d)['x']

    b = create_assignment(x_sol, A_trim)

    output = []
    for i in range(len(b)):
        r = [list(df['person_id'])[b_] for b_ in np.nonzero(b[i])[0]]
        output.append([list(df.person_id)[i], r])

    # make optimal schedule [[person_id, [match_id_1, match_id_2, ...]], ...]
    schedule = nest_answer(output, format_answer(color_graph(build_line_graph(output))))

    # make the document from calculated schedule
    schedule_df = pd.DataFrame(schedule, columns=['person_id', 'match_id'])
    schedule_df['match_id'] = schedule_df.match_id.map(lambda x: x[0: n_meeting])

    # create a full mind-matching dataframe
    mind_matching_df = []
    for i in range(n_meeting):
        schedule_df['match'] = schedule_df.match_id.map(lambda x: x[i])
        match_pairs = list(pd.unique([frozenset((r['person_id'], int(r['match']))) 
                                for _, r in schedule_df.iterrows() if not pd.isnull(r['match'])]))

        r = list(set(schedule_df.person_id) - set(schedule_df['match'].dropna().unique().astype(int)))
        random.shuffle(r)
        match_pairs.extend(list(map(frozenset, zip(r[0:int(len(r)/2)], r[int(len(r)/2):]))))
        match_lookup = [(list(k), v) for v, k in enumerate(match_pairs, start=1)]
        person_lookup = {}
        for k, v in match_lookup:
            person_lookup[k[0]] = k[1]
            person_lookup[k[1]] = k[0]
        match_df = pd.DataFrame(list(chain.from_iterable([[[k[0], v], [k[1], v]] for k, v in match_lookup])), 
                                columns=['person_id', 'table_number'])
        match_df['person_to_meet_id'] = match_df.person_id.map(lambda x: person_lookup[x])
        match_df['timeslot'] = i + 1
        mind_matching_df.append(match_df)
    mind_matching_df = pd.concat(mind_matching_df)


    # create schedule for mind matching
    pages = []
    for person_id, mind_matching_schedule_df in mind_matching_df.groupby('person_id'):
        page = []
        page.extend([
            person_id_map[person_id], 
            person_affil_map[person_id], 
            'RegID: {}'.format(registration_id_map[person_id])
        ])
        page.extend([
            '-------------------',
            'Mind Matching Schedule',
            '-------------------'
        ])
        for _, r in mind_matching_schedule_df.iterrows():
            page.extend([
                'timeslot: {}, table number: {}, mind-match: {} ({})'.\
                format(r['timeslot'], r['table_number'], person_id_map[r['person_to_meet_id']], person_affil_map[r['person_to_meet_id']])
            ])
        pages.append('\n'.join(page))

    # save to word document
    document = Document()
    for page in pages:
        document.add_paragraph(page)
        document.add_page_break()
    document.save('ccn_mindmatch_2019.docx')