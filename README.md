# Paper-Reviewer Matcher

(work in progress)

Package for Paper-Reviewer matching algorithm based on topic modeling.
Algorithm is available to use easily at http://pr.scienceofscience.org/
(implementation based on this [article](http://www.cis.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR08.pdf)).
This package solves problem of assigning paper to reviewers with constrains by solving linear programming problem.
We minimize global distance between papers and reviewers in topic space (e.g. topic can be Principal component,
  Latent Semantic Analysis).

Here is a diagram of problem setup and how we solve the problem.

<img src="figures/problem_setup.png" width="300">

<img src="figures/paper_reviewer_matching.png" width="600">


## Example script

- `ccn_mind_matching.py` contains script for Mind Matching session (match scientists to scientists) for CCN conference
- `ccn_paper_reviewer_matching.py` contains script for matching publications to reviewers for CCN conference, see 
example of CSV files in `data` folder

## Example Usage

I haven't put all functions together in a nice big function. However, here is an
example to solve paper-reviewer assignment problem.

```python
from paper_reviewer_matcher import preprocess, affinity_computation,
                                   create_lp_matrix, linprog, create_assignment
papers = list(map(preprocess, papers)) # list of papers' abstract
reviewers = list(map(preprocess, reviewers)) # list of reviewers' abstract
A = affinity_computation(papers, reviewers,
                         n_components=10, min_df=1, max_df=0.8,
                         weighting='tfidf', projection='pca')
# set conflict of interest by setting A[i, j] to -1000 or lower value
v, K, d = create_lp_matrix(A, min_reviewers_per_paper=0, max_reviewers_per_paper=3,
                              min_papers_per_reviewer=0, max_papers_per_reviewer=3)
x_sol = linprog(v, K.toarray(), d)['x'] # using scipy linprog for python 3
b = create_assignment(x_sol, A) # transform solution to assignment matrix
```

Output `b` is a binary assignment array where rows correspond to papers and
column correspond to reviewers, where row and column _i, j_ correspond to the
assignment of paper _i_ to reviewer _j_. For example, if we want to see paper
in row 0 will be assigned to which reviewers

```python
i = 0 # first paper
print([reviewers[b_] for b_ in np.nonzero(b[i])[0]]) # abstract of reviewers who will review paper 0
```


## Dependencies

- numpy
- scipy
- nltk
- scikit-learn
- [or-tools](https://github.com/google/or-tools) (linear programming solver for python 2.7)

please refer to [Stackoverflow](http://stackoverflow.com/questions/26593497/cant-install-or-tools-on-mac-10-10)
on how to install `or-tools` on MacOSX. I use `pip` to install `protobuf` before installing `or-tools`

```bash
$ pip install protobuf==3.0.0b4
$ pip install ortools
```

for Python 3.6,

```bash
$ pip install --user --upgrade ortools
```


## Members

- Daniel Acuna (corresponding author)
- Titipat Achakulvisut (re-write code)
- Tulakan Ruangrong
- Konrad Kording
