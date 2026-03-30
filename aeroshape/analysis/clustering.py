"""Point distribution (clustering) laws for mesh generation.

Provides functions that map *n* uniformly indexed points to non-uniform
distributions in the parameter interval [0, 1].  Applying these to the
spanwise or chordwise direction concentrates grid points in regions of
high curvature, improving the accuracy of the GVM Divergence-Theorem
volume computation without increasing the total point count.

All distribution functions share the signature ``f(n) -> np.ndarray``
returning an array of *n* values in [0, 1].  For parameterized laws the
module provides factory functions that return a closure with the same
signature.

Typical usage::

    from aeroshape.clustering import cosine, tanh_two_sided

    X, Y, Z = wing.to_vertex_grids(
        num_points_profile=60,
        spanwise_clustering=cosine,
        chordwise_clustering=tanh_two_sided(1.5),
    )

Available laws
--------------
uniform            Even spacing (default baseline).
cosine             Full-cosine — clusters at both ends.
half_cosine_start  Half-cosine — clusters at start (root / LE).
half_cosine_end    Half-cosine — clusters at end (tip / TE).
tanh_one_sided     One-sided tanh (Roberts) — clusters at start.
tanh_two_sided     Two-sided tanh — clusters at both ends.
exponential        Geometric growth — first cell sets the scale.
vinokur            Two-sided stretching — specify first & last cell sizes.

References
----------
Roberts, G. O.  "Computational meshes for boundary layer problems",
    Proceedings of the Second International Conference on Numerical
    Methods in Fluid Dynamics, Lecture Notes in Physics, 1971.
Vinokur, M.  "On one-dimensional stretching functions for finite-
    difference calculations", J. Comput. Phys. 50, 1983.
Thompson, J. F. et al., "Numerical Grid Generation", 1985.
"""

import numpy as np


# ── Simple distribution laws (direct functions) ─────────────────

def uniform(n):
    """Uniform (linear) distribution.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    np.ndarray
        Linearly spaced values in [0, 1].
    """
    return np.linspace(0.0, 1.0, n)


def cosine(n):
    """Full-cosine distribution — clusters at both ends.

    Uses the Chebyshev node distribution:
        t_i = 0.5 * (1 - cos(pi * i / (n-1)))

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    np.ndarray
        Values in [0, 1], dense near 0 and 1.
    """
    if n < 2:
        return np.array([0.0])
    i = np.arange(n)
    return 0.5 * (1.0 - np.cos(np.pi * i / (n - 1)))


def half_cosine_start(n):
    """Half-cosine distribution — clusters at start (t = 0).

    Useful for concentrating spanwise sections near the wing root,
    or chordwise points near the leading edge.

        t_i = 1 - cos(pi/2 * i / (n-1))

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    np.ndarray
        Values in [0, 1], dense near 0.
    """
    if n < 2:
        return np.array([0.0])
    i = np.arange(n)
    return 1.0 - np.cos(0.5 * np.pi * i / (n - 1))


def half_cosine_end(n):
    """Half-cosine distribution — clusters at end (t = 1).

    Useful for concentrating spanwise sections near the wing tip.

        t_i = sin(pi/2 * i / (n-1))

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    np.ndarray
        Values in [0, 1], dense near 1.
    """
    if n < 2:
        return np.array([0.0])
    i = np.arange(n)
    return np.sin(0.5 * np.pi * i / (n - 1))


# ── Parameterized distribution laws (factory functions) ──────────

def tanh_one_sided(beta=1.5):
    """One-sided hyperbolic-tangent clustering (Roberts transformation).

    Concentrates points near the start (t = 0).  Larger *beta* gives
    stronger clustering.

        t_i = 1 - tanh(beta * (1 - eta_i)) / tanh(beta)

    where eta_i = i / (n - 1).

    Parameters
    ----------
    beta : float
        Stretching parameter (> 0).  Typical range: 1.0 to 3.0.

    Returns
    -------
    callable
        Function ``f(n) -> np.ndarray``.
    """
    def _fn(n):
        if n < 2:
            return np.array([0.0])
        eta = np.linspace(0.0, 1.0, n)
        return 1.0 - np.tanh(beta * (1.0 - eta)) / np.tanh(beta)
    return _fn


def tanh_two_sided(beta=1.5):
    """Two-sided hyperbolic-tangent clustering.

    Concentrates points near both ends (t = 0 and t = 1).  Useful
    for spanwise distributions that need resolution at root and tip.

        t_i = 0.5 * (1 + tanh(beta * (2*eta_i - 1)) / tanh(beta))

    Parameters
    ----------
    beta : float
        Stretching parameter (> 0).  Typical range: 1.0 to 3.0.

    Returns
    -------
    callable
        Function ``f(n) -> np.ndarray``.
    """
    def _fn(n):
        if n < 2:
            return np.array([0.0])
        eta = np.linspace(0.0, 1.0, n)
        return 0.5 * (1.0 + np.tanh(beta * (2.0 * eta - 1.0))
                       / np.tanh(beta))
    return _fn


def exponential(ratio=1.2):
    """Geometric (exponential) growth distribution.

    Each successive interval is *ratio* times the previous one.
    Concentrates points at the start for ratio > 1.

    Parameters
    ----------
    ratio : float
        Growth ratio between consecutive intervals (> 0).
        ratio > 1  → dense at start,
        ratio < 1  → dense at end,
        ratio == 1 → uniform.

    Returns
    -------
    callable
        Function ``f(n) -> np.ndarray``.
    """
    def _fn(n):
        if n < 2:
            return np.array([0.0])
        if abs(ratio - 1.0) < 1e-10:
            return np.linspace(0.0, 1.0, n)

        # Geometric series: ds_i = ds_0 * ratio^i
        # Total = ds_0 * (ratio^(n-1) - 1) / (ratio - 1) = 1
        r = float(ratio)
        k = n - 1
        ds0 = (r - 1.0) / (r ** k - 1.0)
        t = np.zeros(n)
        for i in range(1, n):
            t[i] = t[i - 1] + ds0 * r ** (i - 1)
        t[-1] = 1.0  # enforce exact endpoint
        return t
    return _fn


def vinokur(ds_start=0.01, ds_end=0.01):
    """Vinokur two-sided stretching function.

    Specifies the first and last cell sizes (as fractions of the total
    interval length) and computes a smooth stretching that transitions
    between them.  This is the industry-standard method for CFD grids
    requiring controlled cell sizes at both boundaries.

    Parameters
    ----------
    ds_start : float
        First cell size as a fraction of the interval (0, 1).
    ds_end : float
        Last cell size as a fraction of the interval (0, 1).

    Returns
    -------
    callable
        Function ``f(n) -> np.ndarray``.

    Reference
    ---------
    Vinokur, M., J. Comput. Phys. 50 (1983), pp. 215–234.
    """
    def _fn(n):
        if n < 2:
            return np.array([0.0])

        n1 = n - 1
        S = 1.0 / (n1 * np.sqrt(ds_start * ds_end))

        if S <= 1.0 + 1e-10:
            return np.linspace(0.0, 1.0, n)

        # Solve sinh(delta) / delta = S via Newton iteration
        delta = min(np.sqrt(6.0 * (S - 1.0)), 10.0)
        for _ in range(100):
            sd = np.sinh(delta)
            cd = np.cosh(delta)
            f = sd / delta - S
            fp = (cd * delta - sd) / (delta * delta)
            if abs(fp) < 1e-15:
                break
            delta -= f / fp
            delta = max(delta, 1e-6)
            if abs(f) < 1e-12:
                break

        u = np.linspace(0.0, 1.0, n)

        if abs(ds_start - ds_end) < 1e-10:
            # Symmetric: tanh two-sided with computed delta
            t = 0.5 * (1.0 + np.tanh(delta * (u - 0.5))
                        / np.tanh(0.5 * delta))
        else:
            # Asymmetric: blend two one-sided tanh distributions.
            # Solve for individual stretching params via Newton.
            beta_s = _solve_one_sided_beta(n1, ds_start)
            beta_e = _solve_one_sided_beta(n1, ds_end)
            # Forward (dense at start), backward (dense at end)
            t_fwd = np.tanh(beta_s * u) / np.tanh(beta_s)
            t_bwd = 1.0 - np.tanh(beta_e * (1.0 - u)) / np.tanh(beta_e)
            # Linear blend
            t = (1.0 - u) * t_fwd + u * t_bwd

        t[0] = 0.0
        t[-1] = 1.0
        return t
    return _fn


# ── Internal helpers ─────────────────────────────────────────────

def _solve_one_sided_beta(n1, ds_target):
    """Find beta for one-sided tanh clustering with first cell = ds_target.

    For the formula ``t = 1 - tanh(beta*(1 - u)) / tanh(beta)``,
    the first cell is ``ds_0 = 1 - tanh(beta*(1 - 1/n1)) / tanh(beta)``.

    Used internally by :func:`vinokur` for asymmetric stretching.
    """
    avg = 1.0 / n1
    if ds_target >= avg - 1e-10:
        return 0.01  # near-uniform

    u1 = 1.0 - 1.0 / n1   # = (n1 - 1) / n1
    beta = 2.0
    for _ in range(100):
        tb = np.tanh(beta)
        tu = np.tanh(beta * u1)
        val = 1.0 - tu / tb
        err = val - ds_target
        if abs(err) < 1e-12:
            break
        dtb = 1.0 - tb * tb
        dtu = u1 * (1.0 - tu * tu)
        dval = -(dtu * tb - tu * dtb) / (tb * tb)
        if abs(dval) < 1e-15:
            break
        beta -= err / dval
        beta = max(beta, 0.01)
    return beta
