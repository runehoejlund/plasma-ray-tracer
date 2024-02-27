# Demonstration of Metaplectic Geometrical Optics for Reduced Modeling of Plasma Waves
This repository gives a 1D demonstration of the metaplectic geometrical optics (MGO) framework developed by Lopez et al. The theory, details on the numerical implementation and presentation of the results are found in the preprint article on [arXiv:2402.03882](https://arxiv.org/abs/2402.03882). Please refer to the article for references on the MGO theory. This project is originally based on Højlunds Master's thesis project prepared over six months at the Section for Plasma Physics and Fusion Energy, at the Technical University of Denmark, DTU, in partial fulfillment for the degree: Master of Science in Engineering Physics, MSc Eng. The thesis is not published, but has been included in this repo in the [Thesis.pdf](Thesis.pdf) document. However, please note that the article, not the thesis, represents the latest state of the project. This ReadMe covers how to reproduce the main results of the preprint article.

## Getting Started
This project uses Python and iPython notebooks. The necessary python packages are listed in the `requirements.txt` file. To reproduce the main results, I recommend using the Anaconda-distribution of python and creating a virtual environment by running the following commands in a terminal (e.g. from Visual Studio Code):
```
conda create -n mgo python=3.9.12
conda activate mgo
```
Install requirements using pip:
```
pip install -r requirements.txt
```

## Reproduce the Main Results
The main results are the Airy, Weber and X-EBW coupling examples found in the iPython notebooks:
1. `mgo_ex_airy.ipynb`: The Airy Problem (see section V.A. in the preprint article)
2. `mgo_ex_weber.ipynb`: Weber's Equation (see section V.B. in the preprint article)
3. `mgo_ex_X_EBW.ipynb`: X-B Coupling at the upper hybrid layer in a warm plasma (see section V.C. in the preprint article)

## Notes on the Numerical Implementation
There are two numerical implementations of importance in this work. The first is our approach to ray-tracing to obtain a phase space trajectory $z(\boldsymbol{\tau})$ of a family of rays (i.e. to obtain a ray manifold). The second is our implementation of the method of Metaplectic Geometrical Optics (MGO) developed by N. A. Lopez et al. for reconstructing the wave field.

### Ray Tracing
Our Solution for ray tracing is found in the file `trace_ray.py`. We use the initial value problem (IVP) solver from the SciPy Library to solve Hamilton's ray equations. The IVP solver uses a Runge-Kutta scheme. In all examples presented we know the analytic form of the dispersion symbol, $\mathcal{D}(\mathbf{z})$. To obtain the derivatives of $\mathcal{D}(\mathbf{z})$ we use automatic differentiation with PyTorch. Thus, to trace a set of rays, one only needs to write in a dispersion function using the torch libraries' build-in functions. We have created the module `torch_helper.py` which makes it easier to define the dispersion relation, such that it is both differentiable in the PyTorch world and plottable using matplotlib. Examples of applications can be seen in the beginning of the files: `mgo_ex_airy.ipynb`, `mgo_ex_weber.ipynb`, `mgo_ex_X_EBW.ipynb`.

### MGO
Our implementation of MGO is found in the file `mgo.py`. It depends on the following internal modules:
- `finite_diff.py`: Contains functions to differentiate any function defined on a grid through finite differences.
- `util.py`: Contains a few useful utility functions among other things for working with matrices and vectors defined on the ray manifold $\boldsymbol{\tau}$-grid.
- `gauss_freud_quad.py`: Contains a function for integrating $f(x)$ from $0$ to $\infty$ using Gauss Freud Quadrature of order $n$ (currently max 10). See Donnelly et al. (2021).

We experimented with different methods to perform the analytic continuation needed to evaluate the inverse metaplectic transform integral along the steepest descent contour. We ended up using the aaa algorithm from the baryrat Python package. Further details on this and other matters relating to the implementation and results can be found in the preprint article.
