"""Test PPT measurement support in state_exclusion."""

import numpy as np
import pytest

from toqito.state_opt import state_exclusion, state_exclusion_locc
from toqito.states import bell


def _dm(vec):
    """Build a normalized density matrix from a (possibly unnormalized) vector."""
    v = np.array(vec, dtype=complex).reshape(-1, 1)
    return v @ v.conj().T / float(np.linalg.norm(v) ** 2)


# A fixed two-qubit ensemble with a known global < PPT exclusion gap, taken from the
# symmetric-extension exclusion analysis (issue #1523). Computational basis order |00>, |01>, |10>, |11>.
_GAP_VECS = [(0, 1, 1, 1), (0, 0, 1, 1), (1, 0, 1, 1)]


def test_ppt_min_error_primal_matches_dual():
    """For the gap ensemble, the PPT min-error exclusion primal and dual agree (strong duality)."""
    states = [_dm(v) for v in _GAP_VECS]
    probs = [1 / 3] * 3
    primal, _ = state_exclusion(
        states,
        probs,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="primal",
        cvxopt_kktsolver="ldl",
    )
    dual, _ = state_exclusion(
        states,
        probs,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )
    assert np.isclose(primal, dual, atol=1e-4)
    assert np.isclose(primal, 0.08079, atol=1e-3)


def test_ppt_exclusion_bounded_between_global_and_separable():
    """The PPT exclusion value lies between the global and a separable (one-way LOCC) value."""
    states = [_dm(v) for v in _GAP_VECS]
    glob, _ = state_exclusion(states, primal_dual="dual", cvxopt_kktsolver="ldl")
    ppt, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )
    # LOCC measurements are a subset of separable, which are a subset of PPT, so the LOCC exclusion error is
    # a separable-realizable upper bound on the PPT exclusion error: global <= PPT <= SEP <= LOCC.
    locc = state_exclusion_locc(states, dim=[2, 2], reps=3, seed=1)
    assert glob <= ppt + 1e-4
    assert ppt <= locc + 1e-3


def test_ppt_exclusion_strictly_above_global():
    """PPT exclusion is strictly worse (larger error) than global exclusion for the gap ensemble."""
    states = [_dm(v) for v in _GAP_VECS]
    glob, _ = state_exclusion(states, primal_dual="dual", cvxopt_kktsolver="ldl")
    ppt, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual="dual",
        cvxopt_kktsolver="ldl",
    )
    assert ppt > glob + 1e-2


@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_ppt_exclusion_bell_states_perfectly_excludable(primal_dual):
    """The four Bell states are antidistinguishable even under PPT measurements (value ~ 0)."""
    states = [bell(0), bell(1), bell(2), bell(3)]
    val, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        primal_dual=primal_dual,
        cvxopt_kktsolver="ldl",
    )
    assert np.isclose(val, 0.0, atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("primal_dual", ["primal", "dual"])
def test_ppt_exclusion_bell_tensored_with_resource(primal_dual):
    """Bell states tensored with a Bell resource state remain PPT-antidistinguishable (value ~ 0).

    This is the level-1 symmetric-extension (PPT) value referenced in issue #1523; the known value is 0.
    """
    states = [np.kron(bell(i), bell(0)) for i in range(4)]
    val, _ = state_exclusion(
        states,
        measurement="ppt",
        subsystems=[0, 2],
        dimensions=[2, 2, 2, 2],
        primal_dual=primal_dual,
        cvxopt_kktsolver="ldl",
    )
    assert np.isclose(val, 0.0, atol=1e-3)


def test_ppt_exclusion_unambiguous_primal_matches_dual():
    """The unambiguous PPT exclusion primal and dual agree (strong duality)."""
    e_0 = np.array([[1.0], [0.0]])
    e_p = np.array([[1.0], [1.0]]) / np.sqrt(2)
    states = [np.kron(e_0, e_0), np.kron(e_p, e_p)]
    probs = [1 / 2, 1 / 2]
    primal, _ = state_exclusion(
        states,
        probs,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        strategy="unambiguous",
        primal_dual="primal",
        abs_ipm_opt_tol=1e-5,
    )
    dual, _ = state_exclusion(
        states,
        probs,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        strategy="unambiguous",
        primal_dual="dual",
        abs_ipm_opt_tol=1e-5,
    )
    assert np.isclose(primal, dual, atol=1e-3)


def test_ppt_exclusion_unambiguous_never_below_global():
    """The unambiguous PPT exclusion error is never below the global unambiguous exclusion error."""
    e_0 = np.array([[1.0], [0.0]])
    e_p = np.array([[1.0], [1.0]]) / np.sqrt(2)
    states = [np.kron(e_0, e_0), np.kron(e_p, e_p)]
    probs = [1 / 2, 1 / 2]
    glob, _ = state_exclusion(states, probs, strategy="unambiguous", primal_dual="dual", abs_ipm_opt_tol=1e-5)
    ppt, _ = state_exclusion(
        states,
        probs,
        measurement="ppt",
        subsystems=[0],
        dimensions=[2, 2],
        strategy="unambiguous",
        primal_dual="dual",
        abs_ipm_opt_tol=1e-5,
    )
    assert ppt >= glob - 1e-3


@pytest.mark.parametrize("strategy", ["min_error", "unambiguous"])
def test_ppt_exclusion_requires_subsystems_and_dimensions(strategy):
    """Omitting `subsystems` or `dimensions` for PPT measurements raises a ValueError."""
    states = [_dm(v) for v in _GAP_VECS]
    with pytest.raises(ValueError, match="subsystems"):
        state_exclusion(states, measurement="ppt", dimensions=[2, 2], strategy=strategy)
    with pytest.raises(ValueError, match="subsystems"):
        state_exclusion(states, measurement="ppt", subsystems=[0], strategy=strategy)


def test_ppt_exclusion_dimension_product_mismatch():
    """A `dimensions` whose product is not the state dimension raises a ValueError."""
    states = [_dm(v) for v in _GAP_VECS]  # dimension 4
    with pytest.raises(ValueError, match="product"):
        state_exclusion(states, measurement="ppt", subsystems=[0], dimensions=[2, 3])
