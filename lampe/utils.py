r"""General purpose helpers."""

import numpy as np
import os
import torch
import torch.nn as nn

from functools import lru_cache, partial
from itertools import islice, starmap
from torch import Tensor
from torch.optim import Optimizer
from typing import *


def cache(f: Callable = None, /, persist: bool = False) -> Callable:
    r"""Unbounded function cache decorator.

    Wraps a function with a memoizing callable that saves call results, which
    can save time when an expensive function is called several times with the
    same arguments.

    The positional and keyword arguments of :py:`f` must be hashable.

    Arguments:
        f: The function to decorate.
        persist: Whether the cached values persist to disk or not.

    Example:
        >>> @cache
        ... def fib(n):
        ...     return n if n < 2 else fib(n-2) + fib(n-1)
        ...
        >>> fib(42)
        267914296
    """

    if persist:
        try:
            from joblib import Memory
        except ImportError as e:
            print(f"ImportWarning: {e}. Fallback to regular cache.")
        else:
            memory = Memory(os.path.expanduser('~/.cache'), mmap_mode='c', verbose=0)

            if f is None:
                return memory.cache
            else:
                return memory.cache(f)

    d = lru_cache(maxsize=None)

    return d if f is None else d(f)


def vectorize(f: Callable = None, /, **kwargs):
    r"""Convenience vectorization decorator.

    Defines a vectorized function which takes a sequence of objects or NumPy arrays
    as inputs and returns a tuple of NumPy arrays. The vectorized function evaluates
    :py:`f` over successive tuples of the input arrays like the :func:`map` function,
    except it uses the broadcasting rules of NumPy.

    Arguments:
        f: The function to decorate.
        kwargs: Keyword arguments passed to :class:`numpy.vectorize`.

    Example:
        >>> @vectorize(otypes=[float])
        ... def abs(x):
        ...     return x if x > 0 else -x
        ...
        >>> abs(range(-3, 4))
        array([3., 2., 1., 0., 1., 2., 3.])
    """

    if f is None:
        return partial(vectorize, **kwargs)
    else:
        class Vectorize(np.vectorize):
            def _vectorize_call(self, func: Callable, args: List) -> Any:
                if self.signature is not None:
                    return self._vectorize_call_with_signature(func, args)
                elif not args:
                    return func()
                else:
                    ufunc, otypes = self._get_ufunc_and_otypes(func=func, args=args)

                    outputs = ufunc(*args)

                    if ufunc.nout == 1:
                        if otypes[0] == 'O':
                            return outputs
                        else:
                            return np.asanyarray(outputs, dtype=otypes[0])
                    else:
                        return tuple(
                            x if t == 'O' else np.asanyarray(x, dtype=t)
                            for x, t in zip(outputs, otypes)
                        )

        return Vectorize(f, **kwargs)


def broadcast(*tensors: Tensor, ignore: Union[int, List[int]] = None) -> List[Tensor]:
    r"""Broadcasts tensors together.

    The term broadcasting describes how PyTorch treats tensors with different shapes
    during arithmetic operations. In short, if possible, dimensions that have
    different sizes are expanded (without making copies) to be compatible.

    Arguments:
        ignore: The number(s) of dimensions not to broadcast.

    Example:
        >>> x = torch.rand(3, 1, 2)
        >>> y = torch.rand(4, 5)
        >>> x, y = broadcast(x, y, ignore=1)
        >>> x.shape
        torch.Size([3, 4, 2])
        >>> y.shape
        torch.Size([3, 4, 5])
    """

    if type(ignore) is int:
        ignore = [ignore] * len(tensors)
    dims = [t.dim() - i for t, i in zip(tensors, ignore)]

    common = torch.broadcast_shapes(*(
        t.shape[:i]
        for t, i in zip(tensors, dims)
    ))

    return [
        torch.broadcast_to(t, common + t.shape[i:])
        for t, i in zip(tensors, dims)
    ]


def deepto(obj: Any, *args, **kwargs) -> Any:
    r"""Moves and/or casts all tensors referenced in an object, recursively.

    .. warning::
        Unless a tensor already has the correct type and device, the referenced
        tensors are replaced by a copy with the desired properties, which will
        break cross-references. Proceed with caution.

    Arguments:
        obj: An object.
        args: Positional arguments passed to :func:`torch.Tensor.to`.
        kwargs: Keyword arguments passed to :func:`torch.Tensor.to`.

    Example:
        >>> tensors = [torch.arange(i) for i in range(1, 4)]
        >>> deepto(tensors, dtype=torch.float)
        [tensor([0.]), tensor([0., 1.]), tensor([0., 1., 2.])]
    """

    if torch.is_tensor(obj):
        obj = obj.to(*args, **kwargs)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = deepto(value, *args, **kwargs)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = deepto(value, *args, **kwargs)
    elif isinstance(obj, tuple):
        obj = tuple(deepto(value, *args, **kwargs) for value in obj)
    elif hasattr(obj, '__dict__'):
        deepto(obj.__dict__, *args, **kwargs)

    return obj


def starcompose(*fs: Callable) -> Callable:
    r"""Returns the composition :math:`g` of a sequence of functions
    :math:`(f_1, f_2, \dots, f_n)`.

    .. math:: g = f_n \circ \dots \circ f_2 \circ f_1

    If the output :math:`x_i` of the intermediate function :math:`f_i` is a tuple,
    its elements are used as separate arguments for the next function :math:`f_{i+1}`.

    Arguments:
        fs: A sequence of functions :math:`(f_1, f_2, \dots, f_n)`.

    Returns:
        The composition :math:`g`.

    Example:
        >>> g = starcompose(lambda x: x**2, lambda x: x/2)
        >>> g(5)
        12.5
    """

    def g(*x: Any) -> Any:
        for f in fs:
            x = f(*x) if isinstance(x, tuple) else f(x)

        return x

    return g


class GDStep(object):
    r"""Creates a callable that performs gradient descent (GD) optimization steps
    for parameters :math:`\phi` with respect to differentiable loss values.

    The callable takes a scalar loss :math:`l` as input, performs a step

    .. math:: \phi \gets \text{GD}(\phi, \nabla_{\!\phi} \, l)

    and returns the loss, detached from the computational graph. To prevent invalid
    parameters, steps are skipped if not-a-number (NaN) or infinite values are found
    in the gradient. This feature requires CPU-GPU synchronization, which could be a
    bottleneck for some applications.

    Arguments:
        optimizer: An optimizer instance (e.g. :class:`torch.optim.SGD`).
        clip: The norm at which the gradients are clipped. If :py:`None`,
            gradients are not clipped.
    """

    def __init__(self, optimizer: Optimizer, clip: float = None):

        self.optimizer = optimizer
        self.parameters = [
            p
            for group in optimizer.param_groups
            for p in group['params']
        ]
        self.clip = clip

    def __call__(self, loss: Tensor) -> Tensor:
        if loss.isfinite():
            self.optimizer.zero_grad()
            loss.backward()

            if self.clip is None:
                self.optimizer.step()
            else:
                norm = nn.utils.clip_grad_norm_(self.parameters, self.clip)
                if norm.isfinite():
                    self.optimizer.step()

        return loss.detach()


class PlateauDetector(object):
    r"""Creates a plateau detector for online sequences.

    Each time a new value :math:`x_t` is provided, it is compared with the current
    best value :math:`x_b` to determine whether :math:`t` is the new best time step.
    In the minimization mode, :math:`x_t` is considered better if it satisfies

    .. math:: x_t < x_b \, (1 - \tau) ,

    where :math:`\tau \in [0, 1]` is a significance threshold. If it is the case,
    :math:`b` becomes :math:`t`. The sequence is currently at a plateau if :math:`b`
    has not changed for more than :math:`\lambda` patience steps, i.e. if

    .. math:: t - b > \lambda .

    Arguments:
        threshold: The significance threshold :math:`\tau`.
        patience: The patience :math:`\lambda`.
        mode: The improvement mode, either :py:`'min'` or :py:`'max'`.
    """

    def __init__(
        self,
        threshold: float = 1e-2,
        patience: int = 8,
        mode: str = 'min',  # 'max'
    ):
        self.threshold = threshold
        self.patience = patience
        self.mode = mode

        self.sequence = [float('+inf' if mode == 'min' else '-inf')]
        self.best_time = 0

    @property
    def time(self) -> int:
        return len(self.sequence) - 1

    @property
    def best(self) -> float:
        return self.sequence[self.best_time]

    def step(self, value: float) -> None:
        self.sequence.append(value)

        if self.mode == 'min':
            better = value < self.best * (1 - self.threshold)
        else:
            better = value > self.best * (1 + self.threshold)

        if better:
            self.best_time = self.time

    @property
    def plateau(self) -> bool:
        return self.time - self.best_time > self.patience


def gridapply(
    self,
    f: Callable[[Tensor], Tensor],
    bins: Union[int, List[int]],
    bounds: Tuple[Tensor, Tensor],
    batch_size: int = 2**12,  # 4096
) -> Tensor:
    r"""Evaluates a function :math:`f(x)` over a multi-dimensional domain split
    into grid cells. Instead of evaluating the function cell by cell, batches are
    given to the function.

    Arguments:
        f: A function :math:`f(x)`.
        bins: The number(s) of bins per dimension.
        bounds: A tuple of lower and upper domain bounds.
        batch_size: The size of the batches given to the function.

    Example:
        >>> f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> lower, upper = torch.zeros(3), torch.ones(3)
        >>> y = gridapply(f, 50, bounds=(lower, upper))
        >>> y.shape
        torch.Size([50, 50, 50])
    """

    lower, upper = bounds

    # Shape
    dims = len(lower)

    if type(bins) is int:
        bins = [bins] * dims

    # Create grid
    domains = []

    for l, u, b in zip(lower, upper, bins):
        step = (u - l) / b
        dom = torch.linspace(l, u - step, b).to(step) + step / 2.
        domains.append(dom)

    grid = torch.stack(torch.meshgrid(*domains, indexing='ij'), dim=-1)
    grid = grid.reshape(-1, dims)

    # Evaluate f(x) on grid
    y = [f(x) for x in grid.split(batch_size)]

    return torch.cat(y).reshape(*bins, *y.shape[1:])
