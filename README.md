# Noncommutative curves and surfaces in small degree

**C. Fable and T. Banica** (November 2026)

This repository contains the paper (20 pages, PDF and LaTeX source) and the Python programs used for its computations.

## Summary

The free real space $\mathbb R^N_+$ is the abstract space whose coordinates $x_1,\ldots,x_N$ are self-adjoint, and subject to no other relation. We study its hypersurfaces $f=0$ of degree 2, 3 and 4, with their classical, matricial, bounded and unbounded points, and with 3 notions of compactness: matricial compactness, compactness, and archimedeanity.

- **Quadrics.** A free quadric is given by a Hermitian form $H=Q+iP$, with $P$ recording the commutators $i[x_i,x_j]$.
- **Conics.** The free conics are classified, following Ostrovskyĭ and Samoĭlenko, into elliptic conics $z^*z-qzz^*=\varepsilon(1-q)$, hyperbolic conics, and a list of 11 parabolic and flat conics. When $P\neq0$ all the matricial points are classical. The points of the free circle, of the free hyperbola and of the quantum discs are described.
- **Elliptic quadrics.** For $\sum_k(w_k^*w_k-q_kw_kw_k^*)=1$, with $\kappa=\sum_k\max(q_k,0)$, the quadric is always matricially compact, is compact precisely when $\kappa<1$, with $\max_k||w_k||^2=(1-\kappa)^{-1}$, and is archimedean precisely when $\kappa=0$.
- **Cubics.** A free curve $p(x)=g(y)$ is fibered into duals of free products of cyclic groups $\mathbb Z_r*\mathbb Z_s$, and for the elliptic curves $y^2=p(x)$ the group $PSL_2(\mathbb Z)$ appears. The cubic $x^2+y^2+[x,[x,y]]=1$ is compact.
- **Quartics.** A free hypersurface whose top Gram matrix is positive definite is archimedean. Any compact free variety given by finitely many equations is homeomorphic to a quartic. The quartic curves $z^*z=r\,zz^*(1-zz^*)$ contain the dynamics of the logistic map, and fall into uncountably many homeomorphism classes.
- **Symmetries.** $U_n^+$ acts on the isotropic quadrics, and a real form $O_J^+$ acts on the free hyperboloids. The paper ends with a list of open questions.

## Contents

- `014.pdf`, `014.tex`: the paper and its LaTeX source.
- Python programs, each writing a `.log` file, included here:

| File | Content | Section |
|---|---|---|
| `conics.py` | trace identities, points of the free circle and of the free hyperbola, Fock representations, compact cubic | 3, 4, 6 |
| `trees.py` | tree representations of the elliptic quadrics, norms against $(1-\kappa)^{-1}$ | 5 |
| `logistic.py` | cycles of the logistic map, matricial points of the quartics $L_r$, fixed point counts | 7 |

## Running the code

Python 3 with `numpy` is needed. Each program runs from the top folder, in a few seconds, for instance:

```
python3 conics.py
```

