# The logistic quartics z^*z = r zz^*(1 - zz^*).
# The irreducible matricial points of size k come from the k-cycles of F(m) = r m (1 - m):
#   z e_i = sqrt(m_(i+1)) e_(i+1), and z e_(k-1) = lambda sqrt(m_0) e_0, with |lambda| = 1.
# We find the cycles numerically, we check the relation on the corresponding matrices,
# and we print the numbers c_k(r) of k-cycles, which are homeomorphism invariants.
import numpy as np

def cycles(r, k, grid=400001):
    F = lambda m: r * m * (1 - m)
    x = np.linspace(1e-9, 1 - 1e-9, grid)
    y = x.copy()
    for i in range(k):
        y = F(y)
    d = y - x
    roots = []
    for j in np.nonzero(d[:-1] * d[1:] < 0)[0]:
        a, b = x[j], x[j + 1]
        for it in range(60):
            c = (a + b) / 2
            v = c
            for i in range(k):
                v = F(v)
            if (v - c) * d[j] > 0:
                a = c
            else:
                b = c
        roots.append((a + b) / 2)
    found = []
    for m in roots:
        orb = [m]
        for i in range(k - 1):
            orb.append(F(orb[-1]))
        if min(abs(orb[i] - orb[0]) for i in range(1, k)) < 1e-7 if k > 1 else False:
            continue
        if any(abs(min(orb) - min(o)) < 1e-7 for o in found):
            continue
        found.append(orb)
    return found

def error(r, orb, lam=np.exp(0.3j)):
    k = len(orb)
    z = np.zeros((k, k), dtype=complex)
    for i in range(k):
        z[(i + 1) % k, i] = np.sqrt(orb[(i + 1) % k]) * (lam if i == k - 1 else 1)
    A = z @ z.conj().T
    return np.abs(z.conj().T @ z - r * A @ (np.eye(k) - A)).max()

if __name__ == "__main__":
    out = open("logistic.log", "w")
    def say(s):
        print(s); out.write(s + "\n")
    for r in [2.5, 3.2, 3.5, 3.56, 3.7, 3.84, 4.0]:
        counts, err = [], 0.0
        for k in range(1, 9):
            c = cycles(r, k)
            counts.append(len(c))
            for orb in c:
                err = max(err, error(r, orb))
        say("r = %.2f, numbers of k-cycles in (0,1), k = 1,...,8: %s, max error in the relation %.1e" % (r, counts, err))
        P = [2 + sum(d * counts[d - 1] for d in range(2, n + 1) if n % d == 0) for n in range(1, 9)]
        say("          fixed points of F^n in [0,1], n = 1,...,8: %s, log(P_8)/8 = %.4f" % (P, np.log(P[7]) / 8))
    out.close()
