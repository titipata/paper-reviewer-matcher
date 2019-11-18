import re
from glob import glob
import numpy as np
import pandas as pd
import paper_reviewer_matcher as pm
import scipy.sparse as sp
from paper_reviewer_matcher import preprocess, affinity_computation, \
                                   create_lp_matrix, create_assignment
from scipy.sparse import coo_matrix
from ortools.linear_solver import pywraplp
from fuzzywuzzy import fuzz


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


def find_user_ids(authors):
    user_ids = re.findall(r'#(\w+)', authors)
    return [int(idx) for idx in user_ids]


def clean_authors(authors):
    return re.sub(r'#(\w+)', '', authors).replace('()', '')


def create_coi_list(authors, df):
    cois = []
    for i, r in df.iterrows():
        for cl in r['CollaboratorsList']:
            if max([fuzz.ratio(a, cl) for a in authors]) >= 80:
                cois.append(i)
    return cois


def create_coi_author_ids(user_ids, df):
    cois = []
    for i, r in df.iterrows():
        if r['UserID'] in user_ids:
            cois.append(i)
    return cois


def create_assignment_dataframe(b, reviewer_map, paper_id_map, pool_group='a'):
    """
    Get the assignment array, generate assignment dataframe
    """
    assignments = []
    for i in range(len(b)):
        assignments.append([
            paper_id_map[i], [reviewer_map[b_] for b_ in np.nonzero(b[i])[0]]
        ]) 
    assignments_df = pd.DataFrame(assignments, columns=['PaperID', 'UserIDs'])
    n_reviewers = len(assignments_df.UserIDs.iloc[0])
    for c in range(n_reviewers):
        assignments_df['UserID_{}_{}'.format(pool_group, c + 1)] = assignments_df.UserIDs.map(lambda x: x[c])
    return assignments_df.drop('UserIDs', axis=1)


if __name__ == '__main__':
    submission_path, reviewer_a_path, reviewer_b_path = glob('PATH_TO/cosyne-2020/*.csv')
    submission_df = pd.read_csv(submission_path)
    reviewer_a_df = pd.read_csv(reviewer_a_path)
    reviewer_b_df = pd.read_csv(reviewer_b_path)
    submission_df.loc[:, 'keywords'] = submission_df.Keywords.map(lambda x: \
                                                                x.replace('[', '').replace(']', '').replace(',', '').replace('/', ' '))
    reviewer_a_df.loc[:, 'keywords'] = reviewer_a_df.Keywords.fillna('').map(lambda x: x.replace('[', '').replace(']', '').replace(',', '').replace('/', ' '))
    reviewer_b_df.loc[:, 'keywords'] = reviewer_b_df.Keywords.fillna('').map(lambda x: x.replace('[', '').replace(']', '').replace(',', '').replace('/', ' '))
    reviewer_a_df['UserID'] = reviewer_a_df.UserID.astype(int)
    reviewer_b_df['UserID'] = reviewer_b_df.UserID.astype(int)
    reviewer_a_df['FullName'] = reviewer_a_df['FirstName'] + ' ' + reviewer_a_df['LastName']
    reviewer_b_df['FullName'] = reviewer_b_df['FirstName'] + ' ' + reviewer_b_df['LastName']
    submission_df['AuthorIds'] = submission_df.Authors.map(find_user_ids)
    submission_df['AuthorsList'] = submission_df.Authors.map(lambda x: [n.strip() for n in clean_authors(x).split(',')])

    reviewer_a_df['CollaboratorsList'] = reviewer_a_df['Collaborators'].map(lambda x: [n.strip() for n in x.replace(',', ';').split(';') if n is not None])
    reviewer_b_df['CollaboratorsList'] = reviewer_b_df['Collaborators'].map(lambda x: [n.strip() for n in x.replace(',', ';').split(';') if n is not None])
    reviewer_a_df['CollaboratorsList'] = reviewer_a_df['FullName'].map(lambda x: [x]) + reviewer_a_df['CollaboratorsList']
    reviewer_b_df['CollaboratorsList'] = reviewer_b_df['FullName'].map(lambda x: [x]) + reviewer_b_df['CollaboratorsList']
    reviewer_df = pd.concat((reviewer_a_df, reviewer_b_df)).reset_index(drop=True)

    # affinity matrix
    papers = list((submission_df['keywords'] + \
                ' ' + submission_df['Title'] + \
                ' ' + submission_df['Abstract']).map(preprocess))
    reviewers_a = list((reviewer_a_df['keywords'] + \
                        ' ' + reviewer_a_df['SampleAbstract1'].fillna('') + \
                        ' ' + reviewer_a_df['SampleAbstract2'].fillna('')).map(preprocess))
    reviewers_b = list((reviewer_b_df['keywords'] + \
                        ' ' + reviewer_b_df['SampleAbstract1'].fillna('') + \
                        ' ' + reviewer_b_df['SampleAbstract2'].fillna('')).map(preprocess))
    A = affinity_computation(papers, reviewers_a + reviewers_b,
                            n_components=15, min_df=2, max_df=0.85,
                            weighting='tfidf', projection='pca')
    
    # COIs
    cois_ids = submission_df.AuthorIds.map(lambda x: create_coi_author_ids(x, reviewer_df))
    cois = submission_df.AuthorsList.map(lambda x: create_coi_list(x, reviewer_df))
    cois_df = pd.DataFrame(cois + cois_ids, columns=['AuthorsList'])
    for i, r in cois_df.iterrows():
        if len(r['AuthorsList']) > 0:
            for idx in r['AuthorsList']:
                A[i, idx] = -1000

    # assignment
    A_a, A_b = A[:, :len(reviewer_a_df)], A[:, len(reviewer_a_df):]
    v, K, d = create_lp_matrix(A_a, 
                            min_reviewers_per_paper=2, max_reviewers_per_paper=2,
                            min_papers_per_reviewer=10, max_papers_per_reviewer=12)
    x_sol = linprog(v, K, d)['x']
    b_a = create_assignment(x_sol, A_a)

    v, K, d = create_lp_matrix(A_b, 
                            min_reviewers_per_paper=2, max_reviewers_per_paper=2,
                            min_papers_per_reviewer=10, max_papers_per_reviewer=12)
    x_sol = linprog(v, K, d)['x']
    b_b = create_assignment(x_sol, A_b)

    reviewer_a_map = {i: r['UserID'] for i, r in reviewer_a_df.iterrows()}
    reviewer_b_map = {i: r['UserID'] for i, r in reviewer_b_df.iterrows()}
    paper_id_map = {i: r['PaperID'] for i, r in submission_df.iterrows()}

    assignments_a_df = create_assignment_dataframe(b_a, reviewer_a_map,
                                                paper_id_map, 
                                                pool_group='a')
    assignments_b_df = create_assignment_dataframe(b_b, reviewer_b_map, 
                                                paper_id_map,
                                                pool_group='b')

    writer = pd.ExcelWriter('cosyne-2020-match.xlsx', 
                            engine='xlsxwriter')
    assignments_a_df.to_excel(writer, sheet_name='reviewer_pool_a')
    assignments_b_df.to_excel(writer, sheet_name='reviewer_pool_b')
    writer.save()