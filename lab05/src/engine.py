"""A scalar automatic-differentiation engine.

Every Value remembers the Values it was made from and a rule that sends its
gradient back to them. Calling backward() on the final Value walks this graph
in reverse topological order and fills in .grad everywhere.
"""
import math


class Value:
    def __init__(self, data, parents=(), op=""):
        self.data = float(data)
        self.grad = 0.0
        self.parents = parents           # the Values this one was computed from
        self.op = op                     # which operation made it, for printing
        self._backward = lambda: None    # local derivative rule, set by each operation

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self.op}')"

    @staticmethod
    def wrap(x):
        return x if isinstance(x, Value) else Value(x)

    # ------------------------------------------------------------ operations
    def __add__(self, other):
        other = Value.wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def backward():                  # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = Value.wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def backward():                  # d(ab)/da = b, d(ab)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant exponents"
        out = Value(self.data ** k, (self,), f"**{k}")

        def backward():                  # d(a^k)/da = k a^(k-1)
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def backward():                  # 1 where the input was positive, else 0
            self.grad += (self.data > 0) * out.grad
        out._backward = backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")

        def backward():                  # d(e^a)/da = e^a
            self.grad += out.data * out.grad
        out._backward = backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def backward():                  # d(ln a)/da = 1/a
            self.grad += out.grad / self.data
        out._backward = backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), "sigmoid")

        def backward():                  # d(sigmoid)/da = s(1-s)
            self.grad += s * (1.0 - s) * out.grad
        out._backward = backward
        return out

    # Everything below is built from the operations above.
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Value.wrap(other))

    def __truediv__(self, other):
        return self * Value.wrap(other) ** -1

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return Value.wrap(other) - self

    def __rtruediv__(self, other):
        return Value.wrap(other) / self

    # --------------------------------------------------------------- backward
    def topological_order(self):
        """Every node appears after all the nodes it was computed from."""
        order, visited = [], set()
        stack = [(self, False)]          # iterative DFS, so deep graphs do not hit
        while stack:                     # Python's recursion limit
            node, parents_done = stack.pop()
            if parents_done:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for parent in node.parents:
                if parent not in visited:
                    stack.append((parent, False))
        return order

    def backward(self):
        order = self.topological_order()
        self.grad = 1.0                  # d(output)/d(output)
        for node in reversed(order):     # output first, inputs last
            node._backward()             # each rule adds (+=) into its parents


def value_sum(values):
    return sum(values, Value(0.0))


# ----------------------------------------------------------- a tiny network
class Linear:
    def __init__(self, weight, bias):
        """weight: rows = outputs, columns = inputs (same layout as torch.nn.Linear)."""
        self.weight = [[Value(w) for w in row] for row in weight]
        self.bias = [Value(b) for b in bias]

    def __call__(self, x):
        return [value_sum(w * xi for w, xi in zip(row, x)) + b
                for row, b in zip(self.weight, self.bias)]

    def parameters(self):
        return [w for row in self.weight for w in row] + self.bias


class TinyMLP:
    """2 -> 8 -> 1: ReLU hidden layer, sigmoid output."""

    def __init__(self, w1, b1, w2, b2):
        self.hidden = Linear(w1, b1)
        self.out = Linear(w2, b2)

    def __call__(self, x):
        h = [v.relu() for v in self.hidden(x)]
        return self.out(h)[0].sigmoid()

    def parameters(self):
        return self.hidden.parameters() + self.out.parameters()
