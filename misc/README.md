# Miscellaneous
This folder is not essential for the results of the MGO article, but it contains different experiments and unfinished implementations developed as part of the project. We have not documented the files herin in detail, since they are not used directly in the main results of the preprint article. However, below we explain what is contained in a few of the files.

## Other Methods for Analytic Continuation
`diff_func_fitters.py`: This module contains the class `DifferentiableFunc` which represents a function of multiple variables which is differentiable and is equipped with a method for fitting the function to a dataset. This class was used in the thesis project to perform the analytic continuation instead of the aaa method. We experimented with different ways of fitting the data to an analytic function which can be extended to the complex plane. These methods are implemented in the subclasses of the `DifferentiableFunc` class which include:
    1. `NDPolynomial`: Class for representing a polynomial function of multiple variables.
    2. `RationalFunction`: Class for representing a rational function, i.e. a function which is the ratio of two polynomials: $f(x) = P_L(x)/P_M(x)$.
    3. `FiniteDiffFunc1D`: Class for representing function which is differentiated through finite differences.
    4. `SymbolicFunc1D`: Class for representing function which is differentiated through symbolic differentiation with sympy.
    5. `SumFunc`: Class for representing sum of two DifferentiableFunc functions.

The main results of the thesis use the `RationalFunction` or `FiniteDiffFunc1D` and `SumFunc` classes to perform the analytic continuation.

## 3D Implementation
We have begun, but not completed an implementation of MGO in 3D. The file `go_ex_circle.ipynb` performs a field reconstruction within the GO approximation of an azimuthally symmetric plasma density profile. the file `mgo_3D` is our initial crude attempt at implementing the MGO method in 3D. It inspects a profile which has symmetry in the $y$ and $z$-direction and therefore is really a 1D problem. The method reproduces the correct field in this simple case, but currently fails on the more complicated 3D circle example.

- `diff_func_fitters.py`: This module contains the class `DifferentiableFunc` which represents a function of multiple variables which is differentiable and is equipped with a method for fitting the function to a dataset. This class is needed to perform the analytic continuation needed to evaluate the inverse metaplectic transform integral along the steepest descent contour. I have experimented with different ways of fitting the data to an analytic function which can be extended to the complex plane. These methods are implemented in the subclasses of the `DifferentiableFunc` class which include:
    1. `NDPolynomial`: Class for representing a polynomial function of multiple variables.
    2. `RationalFunction`: Class for representing a rational function, i.e. a function which is the ratio of two polynomials: $f(x) = P_L(x)/P_M(x)$.
    3. `FiniteDiffFunc1D`: Class for representing function which is differentiated through finite differences.
    4. `SymbolicFunc1D`: Class for representing function which is differentiated through symbolic differentiation with sympy.
    5. `SumFunc`: Class for representing sum of two DifferentiableFunc functions.

The main results of the thesis use the `RationalFunction` or `FiniteDiffFunc1D` and `SumFunc` classes to perform the analytic continuation.

## Other Technical Notes

### Convention on Vector dimensions

In our 3D GO and MGO examples I've made 3D vectors have dimension 3 as their last dimensions. The first 3 dimensions are reserved for the parametrisation: $(\tau_1, \tau_2, \tau_3) = (t, y_0, z_0)$.

This is incosistent with the previous convention in my torch helper library, where 3D vectors had 3 as their first dimension. Eventually, we should standardize this by choosing 1 convention everywhere. Probably the convention chosen in the articles by Lopez et al is preferable. Importantly, it has consequences for how to calculate inner products (i.e. should we sum over first or last axis?). Therefore, the torch helper module's `inner_product` function is different from the `util` module's inner product function.