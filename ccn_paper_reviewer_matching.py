from glob import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
from paper_reviewer_matcher import preprocess, affinity_computation, \
                                   create_lp_matrix, create_assignment
from scipy.sparse import coo_matrix
from ortools.linear_solver import pywraplp


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


def assign_articles_to_reviewers(article_df, reviewer_df, people_df):
    """
    Perform reviewer-assignment from dataframe of article, reviewer, and people

    Parameters
    ==========
    article_df: a dataframe that has columns `PaperID`, `Title`, `Abstract`, and `PersonIDList`
        where PersonIDList contains string of simicolon separated list of PersonID
    reviewer_df: a dataframe that has columns `PersonID` and `Abstract`
    people_df:  dataframe that has columns `PersonID`, `FullName`

    We assume `PersonID` is an integer

    Output
    ======
    article_assignment_df: an assigned reviewers dataframe, each row of article will have 
        list of reviewers in `ReviewerIDList` column and their name in reviewer_names
    """
    papers = list((article_df['Title'] + ' ' + article_df['Abstract']).map(preprocess))
    reviewers = list(reviewer_df['Abstract'].map(preprocess))

    # Calculate conflict of interest based on co-authors
    coauthors_df = pd.DataFrame([[int(r.PaperID), int(co_author)]
                                for _, r in article_df.iterrows()
                                for co_author in r.PersonIDList.split(';')],
                                columns = ['PaperID', 'PersonID'])
    article_df['paper_id'] = list(range(len(article_df)))
    reviewer_df['person_id'] = list(range(len(reviewer_df)))
    coi_df = coauthors_df.merge(article_df[['PaperID', 'paper_id']], 
                                on='PaperID').merge(reviewer_df[['PersonID', 'person_id']], 
                                on='PersonID')[['paper_id', 'person_id']]
        
    # calculate affinity matrix
    A = affinity_computation(papers, reviewers,
                             n_components=10, min_df=2, max_df=0.8,
                             weighting='tfidf', projection='pca')
    
    # trim distance that are too high
    A_trim = []
    for r in range(len(A)):
        a = A[r, :]
        a[np.argsort(a)[0:200]] = 0
        A_trim.append(a)
    A_trim = np.vstack(A_trim)

    # assign conflict of interest to have high negative cost
    for i, j in zip(coi_df.paper_id.tolist(), coi_df.person_id.tolist()):
        A_trim[i, j] = -1000

    # for CCN case, 
    v, K, d = create_lp_matrix(A_trim, 
                               min_reviewers_per_paper=6, max_reviewers_per_paper=6,
                               min_papers_per_reviewer=4, max_papers_per_reviewer=6)
    x_sol = linprog(v, K, d)['x']
    b = create_assignment(x_sol, A_trim)
    reviewer_ids = list(reviewer_df.PersonID)
    reviewer_name_dict = {r['PersonID']: r['FullName'] for _, r in people_df.iterrows()} # map reviewer id to reviewer name
    assignments = []
    for i in range(len(b)):
        assignments.append([i, 
                            [reviewer_ids[b_] for b_ in np.nonzero(b[i])[0]], 
                            [reviewer_name_dict[reviewer_ids[b_]] for b_ in np.nonzero(b[i])[0]]])
    assignments_df = pd.DataFrame(assignments, columns=['paper_id', 'ReviewerIDList', 'reviewer_names'])
    assignments_df['ReviewerIDList'] = assignments_df.ReviewerIDList.map(lambda e: ';'.join(str(e_) for e_ in e))
    assignments_df['reviewer_names'] = assignments_df.reviewer_names.map(lambda x: ';'.join(x))
    article_assignment_df = article_df.merge(assignments_df, on='paper_id').drop('paper_id', axis=1)
    return article_assignment_df


if __name__ == '__main__':
    CCN_PATH = '/path/to/*.csv'
    article_path, reviewer_path, people_path = [path for path in glob(CCN_PATH) 
                                                if 'CCN' in path and 'fixed' not in path]
    # there is a problem when encoding lines in the given CSV so we have to use ISO-8859-1 instead
    article_df = pd.read_csv(article_path) # has columns `PaperID`, `Title`, `Abstract`, `PersonIDList`
    reviewer_df = pd.read_csv(reviewer_path, encoding="ISO-8859-1") # has columns `PersonID`, `Abstract`
    people_df = pd.read_csv(people_path, encoding="ISO-8859-1") # has columns `PersonID`, `FullName`
    article_assignment_df = assign_articles_to_reviewers(article_df, reviewer_df, people_df)
    article_assignment_df.to_csv('article_assignment.csv', index=False)