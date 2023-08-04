# Metaplectic Geometrical Optics for Ray Tracing in Fusion Plasmas
This is the repository for my Master's thesis project prepared over six months at the Section for Plasma Physics and Fusion Energy, at the Technical University of Denmark, DTU, in partial fulfillment for the degree: Master of Science in Engineering Physics, MSc Eng.

Please see the [Thesis.pdf](Thesis.pdf) document for the actual thesis where much more information on the project is provided. This ReadMe covers how to reproduce the main results.

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
The main results are the Airy, Weber and XB coupling examples found in the iPython notebooks:
1. `mgo_ex_airy.ipynb`: The Airy Problem (see section 5.2 and 7.1 in the thesis)
2. `mgo_ex_weber.ipynb`: Weber's Equation (see section 5.3 and 7.2 in the thesis)
3. `mgo_ex_X_EBW_at_UH.ipynb`: X-B Coupling at the upper hybrid layer in a warm plasma (see section 7.3 in the thesis)

I have begun, but not completed an implementation of MGO in 3D. The file `go_ex_circle.ipynb` performs a field reconstruction within the GO approximation of an azimuthally symmetric plasma density profile. the file `mgo_3D` is my initial attempt at implementing the MGO method in 3D. It inspects a profile which has symmetry in the $y$ and $z$-direction and therefore is really a 1D problem. The method reproduces the correct field in this simple case, but currently fails on the more complicated 3D circle example.

## Notes on the Numerical Implementation
There are two numerical implementations of importance in this work. The first is my approach to ray-tracing to obtain a phase space trajectory $z(\boldsymbol{\tau})$ of a family of rays (i.e. to obtain a ray manifold). The second is my implementation of the method of Metaplectic Geometrical Optics (MGO) developed by N. A. Lopez et al. for reconstructing the wave field.

### Ray Tracing
My Solution for ray tracing is found in the file `trace_ray.py`. It uses automatic differentiation using PyTorch. To trace a set of rays, one only needs to write in a dispersion function using the torch libraries' build-in functions. I have created the module `torch_helper.py` which makes it easier to define the dispersion relation, such that it is both differentiable in the PyTorch world and plottable using matplotlib. Examples of applications can be seen in the beginning of the files: `mgo_ex_airy.ipynb`, `mgo_ex_weber.ipynb`, `mgo_ex_X_EBW_at_UH.ipynb`, `go_ex_circle.ipynb` and `cold-tracer-with-amplitudes.ipynb`.

### MGO
My implementation of MGO is found in the file `mgo.py`. It depends on the following modules:
- `finite_diff.py`: Contains functions to differentiate any function defined on a grid through
finite differences.
- `util.py`: Contains a few useful utility functions among other things for working with matrices and vectors defined on the ray manifold $\boldsymbol{\tau}$-grid.
- `gauss_freud_quad.py`: Contains a function for integrating $f(x)$ from $0$ to $\infty$ using Gauss Freud Quadrature of order $n$ (currently max 10). See Donnelly et al. (2021).
- `diff_func_fitters.py`: This module contains the class `DifferentiableFunc` which represents a function of multiple variables which is differentiable and is equipped with a method for fitting the function to a dataset. This class is needed to perform the analytic continuation needed to evaluate the inverse metaplectic transform integral along the steepest descent contour. I have experimented with different ways of fitting the data to an analytic function which can be extended to the complex plane. These methods are implemented in the subclasses of the `DifferentiableFunc` class which include:
    1. `NDPolynomial`: Class for representing a polynomial function of multiple variables.
    2. `RationalFunction`: Class for representing a rational function, i.e. a function which is the ratio of two polynomials: $f(x) = P_L(x)/P_M(x)$.
    3. `FiniteDiffFunc1D`: Class for representing function which is differentiated through finite differences.
    4. `SymbolicFunc1D`: Class for representing function which is differentiated through symbolic differentiation with sympy.
    5. `SumFunc`: Class for representing sum of two DifferentiableFunc functions.

The main results of the thesis use the `RationalFunction` or `FiniteDiffFunc1D` and `SumFunc` classes to perform the analytic continuation.

### Other technical notes

#### Convention on Vector dimensions

In my 3D GO and MGO examples I've made 3D vectors have dimension 3 as their last dimensions. The first 3 dimensions are reserved for the parametrisation: $(\tau_1, \tau_2, \tau_3) = (t, y_0, z_0)$.

This is incosistent with the previous convention in my torch helper library, where 3D vectors had 3 as their first dimension. Eventually, I think I should standardize this by choosing 1 convention everywhere. Probably the convention chosen in the articles by Lopez et al is preferable. Importantly, it has consequences for how to calculate inner products (i.e. should we sum over first or last axis?). Therefore, the torch helper module's `inner_product` function is different from the `util` module's inner product function.