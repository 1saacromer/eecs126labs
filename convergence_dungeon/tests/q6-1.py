from otter.test_files import test_case
import numpy as np

# This test verifies the running averages for the finite dungeon.

OK_FORMAT = False
name = "q6-1"
points = 6


@test_case(name="finite running average reproducibility", points=2)
def test_finite_running_avg_reproducibility(finite_running_avg):
    """finite_running_avg should produce the same result when the global RNG seed is reset."""
    import numpy as np
    np.random.seed(42)
    Y1 = finite_running_avg()
    np.random.seed(42)
    Y2 = finite_running_avg()
    assert isinstance(Y1, np.ndarray) and isinstance(Y2, np.ndarray), "finite_running_avg() should return a numpy array."
    assert np.array_equal(Y1, Y2), "finite_running_avg() should be reproducible given a fixed seed."


@test_case(name="finite running average shape and final value", points=2)
def test_finite_running_avg_shape_and_final(finite_running_avg):
    """Check the length and final value of the running average for the finite dungeon."""
    import numpy as np
    # Set the seed before calling the student's function
    np.random.seed(42)
    Y = finite_running_avg()
    assert isinstance(Y, np.ndarray), "finite_running_avg() should return a numpy array."
    assert Y.shape == (1000,), "The running average array must have length 1000."
    # Compute expected final running average via a direct simulation
    # Reset the seed so that our manual simulation uses the same random draws
    np.random.seed(42)
    X = np.zeros(1000, dtype=float)
    for n in range(1, 1001):
        p = n ** (-1.3)
        if np.random.rand() < p:
            X[n - 1] = 1.0
    S = np.cumsum(X)
    Y_expected = S / np.arange(1, 1001)
    # Final average should match the direct simulation
    assert abs(float(Y[-1]) - float(Y_expected[-1])) < 1e-6, (
        f"The final running average in the finite dungeon should be {Y_expected[-1]:.6f}."
    )


@test_case(name="finite running average prefix correctness", points=2)
def test_finite_running_avg_prefix(finite_running_avg):
    """
    Compare the first few running average values to a direct simulation of the finite dungeon.
    We manually simulate X_n = 1 with probability n^{-1.3} and compute Y_n = S_n / n.
    """
    import numpy as np
    np.random.seed(42)
    Y_student = finite_running_avg()
    # Reset the seed so that the manual simulation uses the same random draws
    np.random.seed(42)
    # Direct simulation
    X = np.zeros(1000, dtype=float)
    for n in range(1, 1001):
        p = n ** (-1.3)
        if np.random.rand() < p:
            X[n-1] = 1.0
    S = np.cumsum(X)
    Y_expected = S / np.arange(1, 1001)
    # Compare first 10 entries for equality (within tolerance)
    assert np.allclose(Y_student[:10], Y_expected[:10], atol=1e-6), (
        "finite_running_avg() does not correctly compute the running averages for the finite dungeon."
    )