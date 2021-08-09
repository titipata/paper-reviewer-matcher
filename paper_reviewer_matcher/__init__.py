import sys
from .preprocess import preprocess
from .affinity import affinity_computation, create_lp_matrix, create_assignment
from .vectorizer import LogEntropyVectorizer, BM25Vectorizer
try:
    from .lp import linprog
    print("Using Google ortools library for ILP solver.")
except:
    from scipy.optimize import linprog
    print("Using scipy for ILP solver. It may take really long to solve. Please consider install ortool (see README).")
from .mindmatch import perform_mindmatch
