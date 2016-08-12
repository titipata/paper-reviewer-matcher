import sys
from .preprocess import preprocess
from .affinity import affinity_computation, create_lp_matrix, create_assignment
from .vectorizer import LogEntropyVectorizer, BM25Vectorizer

if sys.version_info[0] == 2:
    from .lp import linprog
else:
    from scipy.optimize import linprog
    print('using scipy for linear programming optimization')
