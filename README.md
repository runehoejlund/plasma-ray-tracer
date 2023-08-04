# Plasma Ray Tracer
Repository for my Thesis projects running until august 2023. It is a work in progess.


## Notes

**Convention on Vector dimensions**

In my 3D GO and MGO examples I've made 3D vectors have dimension 3 as their last dimensions. The first 3 dimensions are reserved for the parametrisation: $(\tau_1, \tau_2, \tau_3) = (t, y_0, z_0)$.

**This is incosistent with the previous convention in my torch helper library, where 3D vectors had 3 as their first dimension. Eventually, I think I should standardize this by choosing 1 convention everywhere. Probably Lopez' convention is preferable.**

Importantly, it has consequences for how to calculate inner products (i.e. should we sum over first or last axis?). Therefore, the torch helper module's `inner_product` function is different from the `util` module's inner product function.