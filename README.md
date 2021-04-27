# Sparse network learning with snlpy

Very large and sparse networks appear often in the wild and present unique algorithmic
opportunities and challenges for the practitioner. While in theory, many operations on
sparse networks can be implemented efficiently, this implementation is often
missing in common libraries and is difficult to reliably implement oneself.
In the python world, this is evident just looking for a library method
to perform matrix multiplication on two sparse matrices.

The goal of snlpy is to provide the most efficient possible python implementations
of useful machine learning algorithms on sparse networks. For now, the only
algorithms provided are some common structural node embedding methods as well
as an approximate ppr routine. 

One major goal of snlpy is easy parallelization. This is achieved via the
[numba](https://numba.pydata.org/) package.  Wherever possible, algorithms
execute in parallel and use all available cores. See the documentation for
setting up your environment.

Implementations of the embedding algorithms already exist. These implementations
are based on those found in [karateclub](https://github.com/benedekrozemberczki/KarateClub).
They have been modified to forego compatibility with networkx and to use numba
for efficiency.

## Installation
Simply clone the repo, navigate to the directory and
```
pip install snlpy/
```

## Demo
There is a jupyter notebook outlining the basic features and usage of snlpy
in the demo/ directory.
