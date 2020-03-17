# Paper-Reviewer Matcher

A python package for paper-reviewer matching algorithm based on topic modeling and linear programming. The algorithm is implemented based on this [article](http://www.cis.upenn.edu/~cjtaylor/PUBLICATIONS/pdfs/TaylorTR08.pdf)). This package solves problem of assigning paper to reviewers with constrains by solving linear programming problem. We minimize global distance between papers and reviewers in topic space (e.g. topic modeling can be Principal component, Latent Semantic Analysis (LSA), etc.).

Here is a diagram of problem setup and how we solve the problem.

<img src="figures/problem_setup.png" width="300">

## Mind-Match Command Line

To perform Mind Match, you can clone the repository and run the follows

``` sh
python mindmatch.py data/mindmatch_example.csv --n_match=6 --n_trim=600
```

directly in this path. This should produce a matching output `output_match.csv` in this relative location.

## Example script for the conferences

Here, I include a recent scripts for our Mind Matching session for CCN conference.

- `ccn_mind_matching_2019.py` contains script for Mind Matching session (match scientists to scientists) for [CCN conference](https://ccneuro.org/2018/)
- `ccn_paper_reviewer_matching.py` contains script for matching publications to reviewers for [CCN conference](https://ccneuro.org/2019/), see example of CSV files in `data` folder

The code makes the distance metric of topics between incoming papers with reviewers (for `ccn_paper_reviewer_matching.py`) and
between people with people (for `ccn_mind_matching_2019`). We trim the metric so that the problem is not too big to solve using `or-tools`.
It then solves linear programming problem to assign the best matches which minimize the global distance between papers to reviewers.
After that, we make the output that can be used by the organizers of the CCN conference -- pairs of paper and reviewers or mind-matching
schedule between people to people in the conference.
You can see of how it works below.

<img src="figures/paper_reviewer_matching.png" width="800">

## Dependencies

This can be done by running `pip install -r requirements.txt`.

See `requirements.txt` for all the dependencies. Please refer to [Stackoverflow](http://stackoverflow.com/questions/26593497/cant-install-or-tools-on-mac-10-10) on how to install `or-tools` on MacOSX. You can use `pip` to install `protobuf` before installing `or-tools`

```bash
pip install protobuf==3.0.0b4
pip install ortools
```

for Python 3.6,

```bash
pip install --user --upgrade ortools
```

## Members

- Daniel Acuna (corresponding author)
- Titipat Achakulvisut (re-write code)
- Tulakan Ruangrong
- Konrad Kording
