# Numerical checks for the free conics.
# (1) the identity tr(m Q(x,y)) = 0 with m = i[x,y], used in the rigidity theorem
# (2) the 2 x 2 points of the free circle, with z^2 central, and the disc |z^2| <= 1
# (3) the 2 x 2 points of the free hyperbola uv + vu = 2, over the region st >= 1
# (4) the Fock representation of the quantum discs z^*z - q zz^* = 1 - q, truncated
import numpy as np
rng = np.random.default_rng(0)
out = open("conics.log", "w")
def say(s):
    print(s); out.write(s + "\n")

def herm(k):
    a = rng.normal(size=(k, k)) + 1j * rng.normal(size=(k, k))
    return (a + a.conj().T) / 2

say("(1) traces of m x^2, m (xy+yx), m y^2, m x, m y, with m = i[x,y]")
for k in [2, 3, 5, 8]:
    x, y = herm(k), herm(k)
    m = 1j * (x @ y - y @ x)
    t = [np.trace(m @ a) for a in [x @ x, x @ y + y @ x, y @ y, x, y]]
    say("  k = %d, max |trace| = %.1e, tr(m^2) = %.3f" % (k, max(abs(u) for u in t), np.trace(m @ m).real))

say("(2) free circle, z = [[0,a],[b,0]] with |a|^2 + |b|^2 = 2")
for r in [0.0, 0.3, 0.7, 1.0]:
    a = np.sqrt(1 + np.sqrt(1 - r * r)); b = r / a * np.exp(0.7j)
    z = np.array([[0, a], [b, 0]])
    x = (z + z.conj().T) / 2; y = (z - z.conj().T) / 2j
    e1 = np.abs(x @ x + y @ y - np.eye(2)).max()
    e2 = np.abs(z @ z - a * b * np.eye(2)).max()
    say("  |z^2| = %.2f, error in x^2+y^2=1: %.1e, error in z^2 scalar: %.1e, [x,y] = 0: %s" % (abs(a * b), e1, e2, np.allclose(x @ y, y @ x)))

say("(3) free hyperbola, u^2 = s, v^2 = t, uv + vu = 2")
for s, t in [(1.0, 1.0), (2.0, 0.5), (2.0, 3.0), (0.5, 10.0)]:
    c = 1 / np.sqrt(s * t); d = np.sqrt(1 - c * c)
    u = np.sqrt(s) * np.array([[1, 0], [0, -1]]); v = np.sqrt(t) * np.array([[c, d], [d, -c]])
    say("  s = %.1f, t = %.1f, error %.1e, [u,v] = 0: %s" % (s, t, np.abs(u @ v + v @ u - 2 * np.eye(2)).max(), np.allclose(u @ v, v @ u)))

say("(4) Fock representation, z e_n = sqrt(1 - q^(n+1)) e_(n+1), truncated at size 200")
for q in [-0.9, -0.5, 0.0, 0.5, 0.9]:
    n = 200
    z = np.zeros((n, n))
    for i in range(n - 1):
        z[i + 1, i] = np.sqrt(1 - q ** (i + 1))
    R = z.T @ z - q * z @ z.T - (1 - q) * np.eye(n)
    say("  q = %4.1f, error away from the cutoff %.1e, norm^2 = %.4f, max(1,1-q) = %.4f" % (q, np.abs(R[:n - 1, :n - 1]).max(), np.linalg.norm(z, 2) ** 2, max(1, 1 - q)))
say("(5) compact cubic x^2 + y^2 + [x,[x,y]] = 1, irreducible point of size 2, and random search")
x = np.array([[1, 0], [0, -1]]) / 2; y = np.array([[-1, np.sqrt(2)], [np.sqrt(2), -1]]) / 2
c = x @ y - y @ x
say("  error %.1e, norm of [x,y] = %.4f" % (np.abs(x @ x + y @ y + x @ c - c @ x - np.eye(2)).max(), np.linalg.norm(c, 2)))
rng = np.random.default_rng(0)
mx, my = 0.0, 0.0
for trial in range(2000):
    k = 2 + trial % 3
    a = np.sort(rng.uniform(-1, 1, k)); x = np.diag(a)
    y = rng.normal(size=(k, k)); y = (y + y.T) / 2
    for it in range(200):
        c = x @ y - y @ x
        r = x @ x + y @ y + x @ c - c @ x - np.eye(k)
        if np.abs(r).max() < 1e-13:
            break
        # Newton step in y, with x fixed: dy -> y dy + dy y + [x,[x,dy]]
        L = np.kron(y, np.eye(k)) + np.kron(np.eye(k), y.T) + np.diag(np.subtract.outer(a, a).ravel() ** 2)
        try:
            y = y - np.linalg.solve(L, r.ravel()).reshape(k, k)
        except np.linalg.LinAlgError:
            break
        y = (y + y.T) / 2
    c = x @ y - y @ x
    if np.abs(x @ x + y @ y + x @ c - c @ x - np.eye(k)).max() < 1e-10:
        mx = max(mx, np.linalg.norm(x, 2)); my = max(my, np.linalg.norm(y, 2))
    if trial % 500 == 499:
        say("  %d trials, max norm of x %.4f (bound 1), max norm of y %.4f (bound 2+sqrt(5) = %.4f)" % (trial + 1, mx, my, 2 + np.sqrt(5)))
out.close()
