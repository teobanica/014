# Tree representations of the elliptic free quadrics
#   sum_k ( w_k^* w_k - q_k w_k w_k^* ) = 1
# Each basis vector is a vertex v of a rooted tree, with a label l(v) and a weight h(v).
# The operator w_k sends a vertex of label k to its parent, with coefficient sqrt(h(v)),
# and kills the vertices having other labels. The relation holds if and only if
#   h(v) = 1 + sum over the children c of v of q_{l(c)} h(c)
# The top vertex is its own parent, which makes the representation finite dimensional.
# We check the relation numerically, and compare the norms with the bound 1/(1-kappa).
import numpy as np

def build(q, depth):
    n = len(q)
    S = [k for k in range(n) if q[k] > 0]
    k0 = S[0]
    parent, label, h = [0], [k0], [0.0]      # vertex 0 is the top, with a loop of label k0
    def subtree(par, lab, d):
        v = len(parent)
        parent.append(par); label.append(lab); h.append(0.0)
        s = 1.0
        if d > 0:
            for j in S:
                c = subtree(v, j, d - 1)
                s += q[j] * h[c]
        h[v] = s
        return v
    s = 1.0
    for j in S:
        if j != k0:
            c = subtree(0, j, depth)
            s += q[j] * h[c]
    h[0] = s / (1 - q[k0])
    dim = len(parent)
    w = [np.zeros((dim, dim)) for k in range(n)]
    for v in range(dim):
        w[label[v]][parent[v], v] = np.sqrt(h[v])
    return w

def check(q, depth):
    w = build(q, depth)
    dim = w[0].shape[0]
    R = sum(w[k].T @ w[k] - q[k] * w[k] @ w[k].T for k in range(len(q)))
    err = np.abs(R - np.eye(dim)).max()
    nrm = max(np.linalg.norm(w[k], 2) ** 2 for k in range(len(q)))
    tr = sum(np.trace(w[k].T @ w[k]) * (1 - q[k]) for k in range(len(q))) / dim
    return dim, err, nrm, tr

if __name__ == "__main__":
    out = open("trees.log", "w")
    def say(s):
        print(s); out.write(s + "\n")
    for q in [[0.3, 0.3], [0.45, 0.45], [0.5, 0.5], [0.6, 0.6], [0.2, 0.5, -0.4], [0.4, 0.4, 0.4]]:
        kappa = sum(max(t, 0) for t in q)
        bound = 1 / (1 - kappa) if kappa < 1 else float("inf")
        say("q = %s, kappa = %.2f, bound 1/(1-kappa) = %s" % (q, kappa, bound))
        for depth in [2, 4, 6, 8]:
            if len([t for t in q if t > 0]) ** depth > 3000:
                continue
            dim, err, nrm, tr = check(q, depth)
            say("  depth %d, dim %5d, error %.1e, max norm^2 %10.4f, normalized trace of sum x_i^2 %.6f" % (depth, dim, err, nrm, tr))
    out.close()
