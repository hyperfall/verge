"""L4 plasticity (real): UPGD's guaranteed mechanism + the interference monitor."""
from __future__ import annotations

import numpy as np

from verge.latent import LayerService
from verge.l4_plasticity import (
    SGD,
    UPGD,
    GradientInterferenceMonitor,
    L4Plasticity,
    continual_plasticity_demo,
)


def test_upgd_injects_non_gradient_variability_sgd_cannot():
    # Dohare's whole point: the fix needs a random, non-gradient component. A weight with
    # ZERO gradient is frozen forever under SGD (ossification) but UPGD can still move it.
    w_sgd = [np.array([1.0, 2.0, 3.0])]
    w_upgd = [np.array([1.0, 2.0, 3.0])]
    zero_grad = [np.zeros(3)]
    SGD(lr=0.1).step(w_sgd, zero_grad)
    UPGD(lr=0.1, sigma=0.5, seed=0).step(w_upgd, zero_grad)
    assert np.allclose(w_sgd[0], [1.0, 2.0, 3.0])        # SGD: stuck
    assert not np.allclose(w_upgd[0], [1.0, 2.0, 3.0])   # UPGD: revivable


def test_upgd_protects_high_utility_weights():
    # With noise off, the utility gate alone should move a high-utility weight LESS than a
    # low-utility one under the same gradient.
    opt = UPGD(lr=0.1, sigma=0.0, seed=0)
    params = [np.array([1.0]), np.array([1.0])]
    opt.step(params, [np.array([0.5]), np.array([0.5])])   # init traces
    opt._traces[0][:] = 10.0     # param 0: high utility (important) -> protected
    opt._traces[1][:] = 0.0      # param 1: low utility -> updated
    before = [p.copy() for p in params]
    opt.step(params, [np.array([0.5]), np.array([0.5])])
    move_high = abs(params[0][0] - before[0][0])
    move_low = abs(params[1][0] - before[1][0])
    assert move_high < move_low


def test_interference_monitor_distinguishes_conflict_from_alignment():
    m = GradientInterferenceMonitor(threshold=0.5)
    g = np.array([1.0, 0.0, -1.0])
    assert m.observe(g, -g) > 0.9          # opposed gradients → high interference
    assert m.observe(g, g) < 0.1           # aligned → none


def test_interference_monitor_circuit_breaks():
    m = GradientInterferenceMonitor(threshold=0.5)
    m.observe(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))   # conflict
    assert m.circuit_break()
    assert m.predicted_forgetting() > 0.5


def test_continual_demo_runs_and_reports_dead_fraction():
    out = continual_plasticity_demo(optimizer_name="upgd", tasks=10, steps=10, seed=0)
    assert 0.0 <= out["dead_unit_fraction"] <= 1.0


def test_service_exposes_optimizer_and_monitor():
    L4 = L4Plasticity()
    assert isinstance(L4, LayerService)
    opt = L4.wrap_optimizer(lr=0.05, sigma=0.05)
    assert isinstance(opt, UPGD)
    L4.observe_interference(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    assert L4.health()["built"] is True
