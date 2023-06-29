'''
To use automatic differentiation of the dispersion
function and its' dependencies, all calculations
must be done using torch tensors. This file contains
a wrapper function called `torch_func` which may be
used as a decorator (see example below).

**Example**
```
# Example
@torch_func
def scale_vector(alpha: ('scalar'), a: ('vector', 'detach'), sign: ('any') = 'positive'):
    if sign == 'positive':
        return alpha*a
    else:
        return -alpha*a

scale_vector(2, np.array([[2, 2, 1], [2, 2, 1]]), sign='negative')
```

**Supported Annotations**

The wrapper expects all function parameters to be
annotated; telling us whether each function parameter
is to be parsed to a torch tensor (annotated `('scalar')`),
a 3D torch tensor (annotated `('vector')`)
or is not to be parsed at all (annotated `('any')`).
Please, refer to the example below.

**Detach Annotation**

You can also use annotations to tell that the torch tensor
should also be detached before performing the calculations
(to exclude it from the backward differentation step).

**Note:** The annotations do not set requirements for the
type of the input variables, but instead provides guidence
on how to parse the input before performing the function
call. Thus, the user may give numpy arrays as input for
a function decorated with `@torch_func`.
'''

import torch
import inspect
from warnings import warn

def to_torch(*vars, dtype='torch.FloatTensor', detach=False):
    def convert(var):
        if isinstance(var, torch.Tensor):
            T = var.type(dtype)
        else:
            T = torch.tensor(var).type(dtype)
        
        if detach:
            return T.detach()
        else:
            return T
    
    if len(vars) == 1:
        return convert(vars[0])
    else:
        return (convert(var) for var in vars)

def to_torch_3D(*vars, dtype='torch.FloatTensor', detach=False):
    def reshape_3D(var):
        return var.reshape(3, -1)
    
    if len(vars) == 1:
        return to_torch(reshape_3D(vars[0]), dtype=dtype, detach=detach)
    else:
        return to_torch(*(reshape_3D(var) for var in vars), dtype=dtype, detach=detach)

def torch_func(func):
    def torch_wrap(*args, **kwargs):
        signature = inspect.signature(func)
        assert signature.parameters.keys() == func.__annotations__.keys(), 'Error: All parameters of a torch function should be annotated. Use the annotation \'any\', to avoid passing parameter to a torch tensor.'
        args_keys = [*func.__annotations__.keys()][:len(args)]
        args_with_keys = dict(zip(args_keys, args))
        default_kwargs = { k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty }
        all_kwargs = {**default_kwargs, **args_with_keys, **kwargs}
        try:
            torch_kwargs = {}
            for i, (var_name, annotation) in enumerate(func.__annotations__.items()):
                if 'scalar' in annotation:
                    v = to_torch(all_kwargs[var_name], detach='detach' in annotation)
                elif 'vector' in annotation:
                    v = to_torch_3D(all_kwargs[var_name], detach='detach' in annotation)
                elif 'any' in annotation:
                    v = all_kwargs[var_name]
                else:
                    warn('unsupported annotation: \'' + str(annotation) + '\'. Use the annotation \'any\' to avoid passing a parameter to a torch tensor.')
                torch_kwargs[var_name] = v
            return func(**torch_kwargs)
        except:
            warn('parsing to torch tensors failed for arguments: ')
            print(all_kwargs)
            return func(*args, **kwargs)
    torch_wrap.__annotations__ = func.__annotations__
    return torch_wrap

@torch_func
def inner_product(a: ('vector'), b: ('vector')):
    if len(a.shape) == 2:
        return torch.matmul(torch.t(a), b).diagonal()
    else:
        return torch.dot(a, b)

@torch_func
def angle(a: ('vector'), b: ('vector')):
    return torch.acos(inner_product(a, b)/(torch.norm(a, dim=0)*torch.norm(b, dim=0)))

def grad(f, x, create_graph=True):
    '''Calculates gradient of f with respect to x.
    This functions simply returns first element
    of output of torch.autograd.grad with create_graph=True.
    '''
    return torch.autograd.grad(f, x, create_graph=create_graph)[0]