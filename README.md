# fast_uot
This is a repository which computes Unbalanced Optimal Transport (UOT) in faster settings.
It implements the methods presented in the publication [Faster Unbalanced Optimal Transport: Translation invariant Sinkhorn and 1-D Frank-Wolfe](https://proceedings.mlr.press/v151/sejourne22a/sejourne22a.pdf), published at [AISTATS2022](http://aistats.org/aistats2022/).
We propose two main implementations of UOT: an acceleration of the Sinkhorn algorithm using entropic regularization, and a Frank-Wolfe method to solve unregularized UOT.


## Package requirements
The computation of unbalanced optimal transport is available with NumPy and PyTorch. To run the package *fastuot* and examples you need the following packages.

* numpy
* pytorch
* cvxpy
* scipy
* numba
* matplotlib
* tqdm


## Install the package
First install the required packages using pip or conda. Then clone the package at the location of your choice. In the folder '../fast_uot' do:

        $ python setup.py install


## If you wish to reproduce experiments using biological data
You need to install wot package, we refer to https://broadinstitute.github.io/wot/tutorial/ for details.
The necessary data is available on the above page, or at the link https://drive.google.com/open?id=1E494DhIx5RLy0qv_6eWa9426Bfmq28po
You should download the data, and put these files in a folder located at '../examples/data'.


## Citing
If the methods detailed in this package were useful for you, please cite:
```
@inproceedings{sejourne2022faster,
  title={Faster Unbalanced Optimal Transport: Translation invariant Sinkhorn and 1-D Frank-Wolfe},
  author={S{\'e}journ{\'e}, Thibault and Vialard, Fran{\c{c}}ois-Xavier and Peyr{\'e}, Gabriel},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4995--5021},
  year={2022},
  organization={PMLR}
}
```
