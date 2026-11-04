"""Float32 relaxed Boolean tables, adapted from Dana Edwards's supplied v4 research.

Frozen functions use paper order: bit 3=f(0,0), bit 2=f(0,1), bit 1=f(1,0), bit 0=f(1,1).
Training uses floating point; deployed inference uses integer table lookup only.
"""
import numpy as np
import scipy.sparse as sp

COEFFS = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, -1], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 0],
                   [0, 1, 1, -2], [0, 1, 1, -1], [1, -1, -1, 1], [1, -1, -1, 2], [1, 0, -1, 0], [1, 0, -1, 1],
                   [1, -1, 0, 0], [1, -1, 0, 1], [1, 0, 0, -1], [1, 0, 0, 0]], dtype=np.float32)
class Layer:
    def __init__(self, inputs, width, kind, arity, rng):
        self.kind, self.k = kind, (2 if kind == "choice" else arity)
        self.wires = rng.integers(0, inputs, (self.k, width))
        self.logits = rng.normal(0, 1, (width, 16 if kind == "choice" else 1 << self.k)).astype(np.float32)
        self.scatter = [sp.csr_matrix((np.ones(width, np.float32), (np.arange(width), w)), shape=(width, inputs))
                        for w in self.wires]
        self.m = np.zeros_like(self.logits)
        self.v = np.zeros_like(self.logits)
        self.t = 0
        self.gate_temperature = 1.0

    def frozen(self):
        """Per table: the 2^k entries (index x = sum_i x_i 2^i over the table's inputs in wiring order)."""
        if self.kind == "choice":
            f = self.logits.argmax(1)  # function number: entries f(0,0) f(0,1) f(1,0) f(1,1) as binary digits
            bits = lambda a, b: (f >> (3 - (2 * a + b))) & 1
            return np.stack([bits(0, 0), bits(1, 0), bits(0, 1), bits(1, 1)], axis=1)  # x = a + 2b
        return (self.logits > 0).astype(np.int64)

    def forward(self, h, discrete=False):
        a = [h[:, w] for w in self.wires]
        if self.kind == "choice":
            if discrete:
                p = np.eye(16, dtype=np.float32)[self.logits.argmax(1)]
            else:
                e = np.exp((self.logits - self.logits.max(1, keepdims=True)) / self.gate_temperature)
                p = e / e.sum(1, keepdims=True)
            c = p @ COEFFS
            out = c[:, 0] + c[:, 1] * a[0] + c[:, 2] * a[1] + c[:, 3] * a[0] * a[1]
            return out, (a, p, c)
        theta = (self.logits > 0).astype(np.float32) if discrete else 1 / (1 + np.exp(-self.logits))
        basis = np.ones(a[0].shape + (1,), np.float32)
        for ai in a:  # input i becomes bit i of the entry index
            basis = np.concatenate([basis * (1 - ai)[..., None], basis * ai[..., None]], axis=2)
        out = np.einsum("nwx,wx->nw", basis, theta, optimize=True)
        return out, (a, theta, basis)

    def backward(self, grad_out, cache, need_input_grad):
        a, x1, x2 = cache
        if self.kind == "choice":
            p, c = x1, x2
            grad_c = np.stack([grad_out.sum(0), (grad_out * a[0]).sum(0), (grad_out * a[1]).sum(0),
                               (grad_out * a[0] * a[1]).sum(0)], axis=1)
            grad_p = grad_c @ COEFFS.T
            grad_logits = p * (grad_p - (grad_p * p).sum(1, keepdims=True)) / self.gate_temperature
            grad_a = [grad_out * (c[:, 1] + c[:, 3] * a[1]), grad_out * (c[:, 2] + c[:, 3] * a[0])] \
                if need_input_grad else None
        else:
            theta, basis = x1, x2
            grad_logits = np.einsum("nw,nwx->wx", grad_out, basis, optimize=True) * theta * (1 - theta)
            grad_a = None
            if need_input_grad:
                grad_a = []
                k = self.k
                shaped = theta.reshape((theta.shape[0],) + (2,) * k)  # axes: table, x_{k-1}, ..., x_0
                for i in range(k):
                    others = [j for j in range(k) if j != i]
                    rest = np.ones(a[0].shape + (1,), np.float32)
                    for j in others:
                        rest = np.concatenate([rest * (1 - a[j])[..., None], rest * a[j][..., None]], axis=2)
                    axis = k - i  # axis of x_i in `shaped`
                    diff = np.take(shaped, 1, axis=axis) - np.take(shaped, 0, axis=axis)
                    diff = diff.reshape(theta.shape[0], -1)  # index over the other inputs, bit order preserved
                    grad_a.append(grad_out * np.einsum("nwx,wx->nw", rest, diff, optimize=True))
        return grad_logits, grad_a

    def adam(self, grad, rate):
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad ** 2
        self.logits -= rate * (self.m / (1 - 0.9 ** self.t)) / (np.sqrt(self.v / (1 - 0.999 ** self.t)) + 1e-8)


class Network:
    def __init__(self, inputs, widths, kind, arity, seed):
        rng = np.random.default_rng(seed)
        self.layers, previous = [], inputs
        for width in widths:
            self.layers.append(Layer(previous, width, kind, arity, rng))
            previous = width
        self.rng = rng

    def forward(self, x, discrete=False):
        caches, h = [], x
        for layer in self.layers:
            h, cache = layer.forward(h, discrete)
            caches.append(cache)
        return h, caches

    def loss_and_grads(self, x, y, classes, temperature):
        out, caches = self.forward(x)
        z = out.reshape(len(out), classes, -1).sum(2) / temperature
        z -= z.max(1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
        loss = -np.log(prob[np.arange(len(y)), y] + 1e-12).mean()
        grad_z = prob.copy()
        grad_z[np.arange(len(y)), y] -= 1
        grad_out = np.repeat(grad_z / (len(y) * temperature), out.shape[1] // classes, axis=1).astype(np.float32)
        grads = [None] * len(self.layers)
        for l in reversed(range(len(self.layers))):
            layer = self.layers[l]
            grads[l], grad_a = layer.backward(grad_out, caches[l], l > 0)
            if l:
                grad_out = sum((layer.scatter[i].T @ grad_a[i].T).T for i in range(layer.k))
        return loss, grads

    def scores(self, x, classes, discrete=True, batch=2000):
        parts = []
        for s in range(0, len(x), batch):
            out, _ = self.forward(x[s:s + batch], discrete)
            parts.append(out.reshape(len(out), classes, -1).sum(2))
        return np.concatenate(parts)


def gradient_check(kind, arity, seed=0):
    """Largest absolute difference between the analytic gradient and central differences, on a small network."""
    rng = np.random.default_rng(seed)
    net = Network(12, [10, 10, 6], kind, arity, seed)
    for layer in net.layers:
        layer.logits = layer.logits.astype(np.float64)
    x = (rng.random((7, 12)) > 0.5).astype(np.float64)
    y = rng.integers(0, 2, 7)

    def loss():
        out, _ = net.forward(x)
        z = out.reshape(len(out), 2, -1).sum(2) / 3.0
        z -= z.max(1, keepdims=True)
        return -np.log((np.exp(z) / np.exp(z).sum(1, keepdims=True))[np.arange(7), y]).mean()

    _, grads = net.loss_and_grads(x, y, 2, 3.0)
    worst = 0.0
    for l, layer in enumerate(net.layers):
        for _ in range(6):
            i, j = rng.integers(0, layer.logits.shape[0]), rng.integers(0, layer.logits.shape[1])
            old = layer.logits[i, j]
            layer.logits[i, j] = old + 1e-5
            up = loss()
            layer.logits[i, j] = old - 1e-5
            down = loss()
            layer.logits[i, j] = old
            worst = max(worst, abs((up - down) / 2e-5 - grads[l][i, j]))
    return worst

