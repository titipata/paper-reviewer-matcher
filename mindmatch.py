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
from paper_reviewer_matcher.affinity import compute_affinity
import numpy as np
import pandas as pd
from docopt import docopt
from paper_reviewer_matcher import (
    preprocess,
    compute_affinity,
    perform_mindmatch,
    compute_conflicts
)


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

    # create affinity matrix and compute conflicts
    persons_1 = list(map(preprocess, list(df['abstracts'])))
    persons_2 = list(map(preprocess, list(df['abstracts'])))
    A = compute_affinity(
        persons_1, persons_2,
        n_components=30, min_df=3, max_df=0.85,
        weighting='tfidf', projection='pca'
    )
    print('Compute conflicts... (this may take a bit)')
    cois = compute_conflicts(df, ratio=85)
    print('Done computing conflicts!')

    # perform mindmatching
    b = perform_mindmatch(A, n_trim=n_trim, n_match=n_match, cois=cois)

    if (b.sum() != 0):
        output = []
        user_ids_map = {ri: r['user_id'] for ri, r in df.iterrows()}
        for i in range(len(b)):
            match_ids = [str(user_ids_map[b_]) for b_ in np.nonzero(b[i])[0]]
            output.append({
                'user_id': user_ids_map[i],
                'match_ids': ';'.join(match_ids)
            })
        output_df = pd.DataFrame(output)
        output_df.to_csv(output_filename, index=False)
        print("Successfully save the output match to {}".format(output_filename))
    else:
        print("Cannot solve the problem")