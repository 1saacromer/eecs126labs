from otter.test_files import test_case
import numpy as np

# This test verifies the running averages for the cursed dungeon.

OK_FORMAT = False
name = "q6-4"
points = 6

@test_case(name="cursed running average reproducibility", points=2)
def test_cursed_running_avg_reproducibility(cursed_running_avg):
    """cursed_running_avg should produce the same result when the global RNG seed is reset."""
    import numpy as np
    np.random.seed(42)
    Y1 = cursed_running_avg()
    np.random.seed(42)
    Y2 = cursed_running_avg()
    assert isinstance(Y1, np.ndarray) and isinstance(Y2, np.ndarray), "cursed_running_avg() should return a numpy array."
    assert np.array_equal(Y1, Y2), "cursed_running_avg() should be reproducible given a fixed seed."

@test_case(name="cursed running average shape and final value", points=2)
def test_cursed_running_avg_shape_and_final(cursed_running_avg):
    """Check the length and final value of the running average for the cursed dungeon."""
    import numpy as np
    # Set the seed before calling the student's function
    np.random.seed(42)
    Y = cursed_running_avg()
    assert isinstance(Y, np.ndarray), "cursed_running_avg() should return a numpy array."
    assert Y.shape == (1000,), "The running average array must have length 1000."
    # Compute expected final running average via a direct simulation of the dependent process
    # Reset the seed so that our manual simulation uses the same random draws
    np.random.seed(42)
    X = np.zeros(1000, dtype=float)
    prev_success = 0
    # Define the dependence: after a success, multiply p_n by 2; after a failure, halve p_n
    for n in range(1, 1001):
        base = 1.0 / n
        if prev_success == 1:
            p = min(2.0 * base, 1.0)
        else:
            p = 0.5 * base
        x = 1.0 if np.random.rand() < p else 0.0
        X[n - 1] = x
        prev_success = int(x)
    S = np.cumsum(X)
    Y_expected = S / np.arange(1, 1001)
    # Final average should match the direct simulation
    assert abs(float(Y[-1]) - float(Y_expected[-1])) < 1e-6, (
        f"The final running average in the cursed dungeon should be {Y_expected[-1]:.6f}."
    )

@test_case(name="cursed running average prefix correctness", points=2)
def test_cursed_running_avg_prefix(cursed_running_avg):
    """
    Compare the first few running average values to a direct simulation of the cursed dungeon.
    We implement the dependence mechanism: p_n = 1/n scaled by 1.5 on a success and 0.5 on a failure.
    """
    import numpy as np
    np.random.seed(42)
    Y_student = cursed_running_avg()
    # Reset the seed so that the manual simulation uses the same random draws
    np.random.seed(42)
    # Direct simulation using the specified dependence mechanism
    X = np.zeros(1000, dtype=float)
    prev_success = 0
    for n in range(1, 1001):
        base = 1.0 / n
        if prev_success == 1:
            p = min(2.0 * base, 1.0)
        else:
            p = 0.5 * base
        x = 1.0 if np.random.rand() < p else 0.0
        X[n-1] = x
        prev_success = int(x)
    S = np.cumsum(X)
    Y_expected = S / np.arange(1, 1001)
    # Compare first 10 entries for equality (within tolerance)
    assert np.allclose(Y_student[:10], Y_expected[:10], atol=1e-6), (
        "cursed_running_avg() does not correctly compute the running averages for the cursed dungeon."
    )
