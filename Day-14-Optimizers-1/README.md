# Day 14  - Momentum & Nesterov Momentum

## Why Momentum Exists

Picture a loss surface shaped like a ravine: steep walls on one axis, a nearly flat floor on the other. Plain SGD takes its step straight along the current gradient, so on the steep axis it overshoots back and forth, zigzagging across the ravine, while on the flat axis it barely advances at all. Momentum fixes this by keeping a running velocity built from past gradients instead of reacting only to the current one. The zigzagging components on the steep axis point in alternating directions and cancel out in that running average, while the small, consistent components along the flat floor accumulate and build speed. Same information, aggregated differently, with the aggregation itself being the entire mechanism.

## Classical Momentum

$$v_t = \beta v_{t-1} + g_t \qquad w_t = w_{t-1} - \eta\, v_t$$

`g_t` is the gradient at the current weights, `η` is the learning rate, and `β` (commonly 0.9) is a memory coefficient: `β = 0` recovers plain SGD exactly, since `v_t` then equals `g_t` every step and the momentum term contributes nothing.

**Why `β` matters quantitatively, not just qualitatively.** Under a steady, unchanging gradient `g`, the velocity recurrence `v_t = β v_{t-1} + g` converges to a fixed point: `v_ss = g / (1 - β)`. That denominator is the entire story. At `β = 0.9`, `1/(1-0.9) = 10`, so a consistent gradient direction gets amplified 10x by the time the velocity settles. At `β = 0.99`, that amplification jumps to 100x. Momentum doesn't just "help a bit" with consistent directions, it compounds them by an amount that grows fast as `β` approaches 1.

## Nesterov Momentum

The core idea: evaluate the gradient not at the current weights, but at the point momentum is already about to carry you to, the "look-ahead" point. Classical momentum reacts to where you are; Nesterov reacts to where you're headed.

$$g_t = \nabla L\!\left(w_{t-1} - \eta \beta\, v_{t-1}\right) \qquad v_t = \beta v_{t-1} + g_t \qquad w_t = w_{t-1} - \eta\, v_t$$

The only change from classical momentum is where the gradient gets evaluated, at the look-ahead point `w_{t-1} - ηβv_{t-1}` instead of at `w_{t-1}` directly. Everything else about the update is identical.

## The Experiment

A ravine loss surface built directly from Day 14's intuition: `0.5(x² + 25y²)`, a 25:1 curvature ratio between the flat `x` axis and the steep `y` axis.

```python
A = np.array([1.0, 25.0])

def loss(w):
    return 0.5 * np.sum(A * w**2)

def grad(w):
    return A * w
```

```python
def classical(w0, lr, beta, steps):
    """v = b*v + g(w);  w = w - lr*v"""
    w, v = w0.copy(), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        v = beta * v + grad(w)
        w = w - lr * v
        path.append(w.copy())
    return np.array(path)
```

```python
def nesterov_lookahead(w0, lr, beta, steps):
    """Original form: gradient at the look-ahead point w - lr*b*v."""
    w, v = w0.copy(), np.zeros_like(w0)
    path = [w.copy()]
    for _ in range(steps):
        g = grad(w - lr * beta * v)       # <- the only change vs classical
        v = beta * v + g
        w = w - lr * v
        path.append(w.copy())
    return np.array(path)
```

```python
w0 = np.array([10.0, 1.0])
lr, beta, steps = 0.05, 0.9, 150

cl = classical(w0, lr, beta, steps)
nl = nesterov_lookahead(w0, lr, beta, steps)

print(f"Classical momentum : {steps_to(cl)} steps to loss < 1e-3")
print(f"Nesterov           : {steps_to(nl)} steps to loss < 1e-3")
```

```text
Classical momentum : 78 steps to loss < 1e-3
Nesterov           : 37 steps to loss < 1e-3
```

A second experiment sweeps the learning rate from 0.005 to 0.20 across 80 points, running each method for 600 steps and recording the first `lr` where it fails to converge.

```python
lrs = np.linspace(0.005, 0.20, 80)
sweep_cl = [steps_to(classical(w0, l, beta, 600)) for l in lrs]
sweep_nl = [steps_to(nesterov_lookahead(w0, l, beta, 600)) for l in lrs]
```

```text
First lr where classical fails to converge: 0.1531
First lr where Nesterov fails to converge:  0.0544
```

| | Classical | Nesterov |
|---|---|---|
| Steps to loss < 1e-3 (lr=0.05) | 78 | 37 |
| Largest stable learning rate | ~0.153 | ~0.054 |

Two separate results sitting in that table, easy to conflate into one. Within the learning rate both methods can actually use, Nesterov converges roughly twice as fast and damps the zigzag oscillation harder than classical momentum does. But Nesterov's stable range is narrower, not wider: it diverges at a learning rate classical momentum handles without trouble. Faster inside its working range and more fragile outside it, both true at once.

## Stability Limits

For a quadratic loss with curvature `λ` along the steepest direction, each method has a closed-form bound on the learning rate before the update diverges:

| Method | Stable if |
|---|---|
| Plain SGD | `ηλ < 2` |
| Classical momentum | `ηλ < 2(1+β)` |
| Nesterov | `ηλ < 2(1+β) / (1+2β)` |

With `λ = 25` (the steep axis of the experiment's ravine) and `β = 0.9`:

$$\eta_{SGD} < \frac{2}{25} = 0.08 \qquad \eta_{classical} < \frac{2(1.9)}{25} = 0.152 \qquad \eta_{nesterov} < \frac{2(1.9)}{2.8 \times 25} = 0.0543$$

These match the swept values (0.1531, 0.0544) closely, the small gap is grid resolution in the sweep, not a mismatch in the formula. The ordering here is the mechanism behind the "faster but narrower" result above: classical momentum's bound, `2(1+β)`, is strictly larger than plain SGD's bound of `2` for any `β > 0`, so **momentum tolerates larger learning rates than plain SGD on this surface, not smaller ones**. It doesn't trade stability for speed against the baseline. Nesterov's extra `(1+2β)` term in the denominator is what shrinks its bound below classical momentum's, that's where its narrower stable range actually comes from, not from momentum itself being destabilizing.

## Real-World Use

- **ResNet / ImageNet training recipe**: SGD with momentum 0.9, learning rate 0.1 with step decay, weight decay 1e-4. One of the most reused optimizer configurations in vision research.
- **CNN fine-tuning for defect detection**: learning rate around 1e-3, momentum 0.9, with a smaller learning rate applied to the pretrained layers than to newly added ones, the usual fine-tuning pattern of protecting what's already learned while letting new layers adapt faster.
- **Why SGD+momentum over Adam, when it's chosen**: SGD with momentum, well-tuned, often generalizes better on vision tasks than Adam, and it uses less memory, one velocity buffer per weight versus Adam's two (first and second moment estimates). The cost is more sensitivity to the learning rate choice, which is exactly the tradeoff the stability bounds above describe concretely instead of just naming.
- **Nesterov on real networks**: the gain over classical momentum tends to be small and noisy on real architectures, worth treating as a low-cost option rather than a guaranteed win. The 2:1 speed advantage in this experiment is a property of this specific quadratic ravine, not a number to expect from a real network's much more complicated loss surface.

## Failure Modes

- **Loss oscillates, then slowly settles.** `β` is too high for the learning rate in use. The stability table above gives the precise boundary; this symptom is what approaching that boundary from the stable side looks like in practice.
- **Loss overshoots right after a learning rate drop.** The velocity buffer still holds momentum built up under the old, larger learning rate, and that momentum takes a few steps to fade under the new recurrence. Not a bug, an expected transient.
- **Momentum doesn't fix a bad learning rate.** It smooths the path the optimizer takes, it doesn't change what learning rate is actually appropriate for the loss surface. A learning rate that's fundamentally too large for the curvature will still diverge, just with a different, momentum-shaped trajectory on the way there.