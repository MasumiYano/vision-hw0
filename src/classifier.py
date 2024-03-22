# Importing from library
import math
import numpy as np

# Importing from src
from src import matrix, data


def active_matrix(m: np.ndarray, a: str) -> None:
    if a == 'RELU':
        np.maximum(m, 0, out=m)
    elif a == 'LOGISTIC':
        np.negative(m, out=m)
        np.exp(m, out=m)
        np.add(1, m, out=m)
        np.divide(1, m, out=m)
    elif a == 'LRELU':
        m[:] = np.where(m > 0, m, 0.01 * m)
    elif a == 'SOFTMAX':
        m -= np.max(m, axis=1, keepdims=True)
        np.exp(m, out=m)
        sum_exp = np.sum(m, axis=1, keepdims=True)
        m /= sum_exp


def gradient_matrix(m: np.ndarray, a: str, d: np.ndarray) -> None:
    if a == 'LINEAR' or a == 'SOFTMAX':
        return
    if a == 'RELU':
        d[:] = np.where(m > 0, 1, 0)
    elif a == 'LRELU':
        d[:] = np.where(m > 0, 1, 0.1)
    elif a == 'LOGISTIC':
        d[:] = d * m * (1 - m)


def forward_layer(l: dict, input: np.ndarray) -> np.ndarray:
    l['in'] = input
    out: np.ndarray = np.dot(l['in'], l['w'])
    active_matrix(out, l['activation'])
    l['out'] = out
    return out


def backward_layer(l: dict, delta: np.ndarray) -> np.ndarray:
    """
    1.4.1
    delta is dL/dy - derivative of the loss w.r.t. output of the layer.
    TODO: modify it in place to be dL/d(xw)
    """
    gradient_matrix(l['out'], l['activation'], delta)

    # 1.4.2
    # TODO: then calculate dL/dw and save it in l->dw
    l['dw'] = np.dot(l['in'].T, delta)

    # 1.4.3
    # TODO: finally, calculate dL/dx and return it
    dL_dx = np.dot(delta, l['w'].T)

    return dL_dx


def update_layer(l: dict, rate: float, momentum: float, decay: float) -> None:
    # TODO: Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}. Save it to l->v
    l['v'] = l['dw'] - decay * l['w'] + momentum * l['v']

    # Update l-> w
    l['w'] = l['w'] + rate * l['v']


def make_layer(input: int, output: int, activation: str) -> dict:
    l: dict = {
        'in': matrix.make_matrix(1, 1),  # input to a layer
        'out': matrix.make_matrix(1, 1),  # output of the layer after weights and activation applied.
        'w': matrix.random_matrix(input, output, math.sqrt(2 / input)),  # weight of the layer
        'v': matrix.make_matrix(input, output),  # momentum
        'dw': matrix.make_matrix(input, output),  # gradient of the loss w.r.t. weights w
        'activation': activation  # Activation the layer uses
    }
    return l


def forward_model(m: dict, X: np.ndarray) -> np.ndarray:
    for layer in m['layers']:
        X = forward_layer(layer, X)
    return X


def backward_model(m: dict, dL: np.ndarray) -> None:
    d: np.ndarray = matrix.copy_matrix(dL)
    for i in reversed(range(len(m['layers']))):
        layer = m['layers'][i]
        d = backward_layer(layer, d)


def update_model(m: dict, rate: float, momentum: float, decay: float) -> None:
    for layer in m['layers']:
        update_layer(layer, rate, momentum, decay)


def max_index(a: np.ndarray) -> int:
    return np.argmax(a)


def accuracy_model(m: dict, d: dict) -> float:
    p: np.ndarray = forward_model(m, d['X'])
    correct: int | float = sum(max_index(d['y'][i]) == max_index(p[i]) for i in range(d['y'].shape[0]))
    return correct / d['y'].shape[0]


def cross_entropy_loss(y: np.ndarray, p: np.ndarray) -> float:
    return -np.sum(y * np.log(p + 1e-9)) / y.shape[0]


def train_model(m: dict, d: dict, batch: int, iters: int, rate: float, momentum: float, decay: float) -> None:
    for e in range(iters):
        b = data.random_batch(d, batch)
        p = forward_model(m, b['X'])
        print(f"{e:06d}: Loss: {cross_entropy_loss(b['y'], p)}")
        dL = matrix.axpy_matrix(-1, p, b['y'])
        backward_model(m, dL)
        update_model(m, rate / batch, momentum, decay)
