#!/usr/bin/env python

"""MindMatch: a script for matching people to people in the conference
Run the script here since 

Usage:
  mindmatch.py PATH [--n_match=<n_match>] [--n_trim=<n_trim>] [--output=<output>]
  mindmatch.py [-h | --help]
  mindmatch.py [-v | --version]

Arguments:
  PATH                  Path to a CSV file, 
                        a file need to have ('user_id', 'fullname', 'abstracts', 'conflicts') in the header

Options:
  -h --help             Show documentation helps
  --version             Show version
  --n_match=<n_match>   Number of match per user
  --n_trim=<n_trim>     Trimming parameter for distance matrix, increase to reduce problem size
  --output=<output>     Output CSV file contains 'user_id' and 'match_ids' which has match ids with ; separated
"""

import os
import sys

import numpy as np
import pandas as pd
from docopt import docopt

from ortools.linear_solver import pywraplp
from paper_reviewer_matcher import preprocess, affinity_computation, \
                                   create_lp_matrix, create_assignment
from fuzzywuzzy import fuzz
from tqdm import tqdm

def linprog(f, A, b):
    """
    Solve the following linear programming problem
            maximize_x (f.T).dot(x)
            subject to A.dot(x) <= b
    where   A is a sparse matrix (coo_matrix)
            f is column vector of cost function associated with variable
            b is column vector
    """

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


if __name__ == "__main__":
    arguments = docopt(__doc__, version='MindMatch 0.1.dev')

    file_name = arguments['PATH']
    df = pd.read_csv(file_name).fillna('')
    assert 'user_id' in df.columns, "CSV file must have ``user_id`` in the columns"
    assert 'fullname' in df.columns, "CSV file must have ``fullname`` in the columns"
    assert 'abstracts' in df.columns, "CSV file must have ``abstracts`` in the columns"
    assert 'conflicts' in df.columns, "CSV file must have ``conflicts`` in the columns"
    print("Number of people in the file = {}".format(len(df)))

    n_match = arguments.get('--n_match')
    if n_match is None:
        n_match = 6
        print('<n_match> is set to default for 6 match per user')
    else:
        n_match = int(n_match)
        print('Number of match is set to {}'.format(n_match))
    assert n_match >= 2, "You should set <n_match> to be more than 2"
    
    n_trim = arguments.get('--n_trim')
    if n_trim is None:
        n_trim = 0
        print('<n_trim> is set to default, this will take very long to converge for a large problem')
    else:
        n_trim = int(n_trim)
        print('Trimming parameter is set to {}'.format(n_trim))

    output_filename = arguments.get('output')
    if output_filename is None:
        output_filename = 'output_match.csv'

    # create assignment matrix
    persons_1 = list(map(preprocess, list(df['abstracts'])))
    persons_2 = list(map(preprocess, list(df['abstracts'])))
    A = affinity_computation(persons_1, persons_2,
                             n_components=30, min_df=3, max_df=0.85,
                             weighting='tfidf', projection='pca')
    A[np.arange(len(A)), np.arange(len(A))] = -1000  # set diagonal to prevent matching with themselve

    print('Compute conflicts... (this may take a bit)')
    cois = compute_conflicts(df)
    A[cois] = -1000
    print('Done computing conflicts!')

    # trimming affinity matrix to reduce the problem size
    if n_trim != 0:
        A_trim = []
        for r in range(len(A)):
            a = A[r, :]
            a[np.argsort(a)[0:n_trim]] = 0
            A_trim.append(a)
        A_trim = np.vstack(A_trim)
    else:
        A_trim = A

    print('Solving a matching problem...')
    v, K, d = create_lp_matrix(A_trim, 
                               min_reviewers_per_paper=n_match, max_reviewers_per_paper=n_match,
                               min_papers_per_reviewer=n_match, max_papers_per_reviewer=n_match)
    x_sol = linprog(v, K, d)['x']
    b = create_assignment(x_sol, A_trim)
    if (b.sum() == 0):
        print('Seems like the problem does not converge, try reducing <n_trim> but not too low!')
    else:
        print('Successfully assigned all the match!')

    if (b.sum() != 0):
        output = []
        user_ids = list(df['user_id'])
        for i in range(len(b)):
            match_ids = [str(user_ids[b_]) for b_ in np.nonzero(b[i])[0]]
            output.append({
                'user_id': user_ids[i],
                'match_ids': ';'.join(match_ids)
            })
        output_df = pd.DataFrame(output)
        output_df.to_csv(output_filename, index=False)
        print('Successfully save the output match to {}'.format(output_filename))