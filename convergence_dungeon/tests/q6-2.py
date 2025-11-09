from otter.test_files import test_case
import numpy as np

# This test verifies the running averages for the endless dungeon.


OK_FORMAT = False
name = "q6-2"
points = 4


@test_case(name="endless running average reproducibility", points=2)
def test_endless_running_avg_reproducibility(endless_running_avg):
    """endless_running_avg should produce the same result when the global RNG seed is reset."""
    import numpy as np
    np.random.seed(42)
    Y1 = endless_running_avg()
    np.random.seed(42)
    Y2 = endless_running_avg()
    assert isinstance(Y1, np.ndarray) and isinstance(Y2, np.ndarray), "endless_running_avg() should return a numpy array."
    assert np.array_equal(Y1, Y2), "endless_running_avg() should be reproducible given a fixed seed."




@test_case(name="endless running average prefix correctness", points=2)
def test_endless_running_avg_prefix(endless_running_avg):
    """
    Compare the first few running average values to a direct simulation of the endless dungeon.
    We manually simulate X_n = 1 with probability 1/n and compute Y_n = S_n / n.
    """
    import numpy as np
    np.random.seed(42)
    Y_student = endless_running_avg()
    # Reset the seed so that the manual simulation uses the same random draws
    np.random.seed(42)
    # Direct simulation
    X = np.zeros(1000, dtype=float)
    for n in range(1, 1001):
        p = 1.0 / n
        if np.random.rand() < p:
            X[n-1] = 1.0
    S = np.cumsum(X)
    Y_expected = S / np.arange(1, 1001)
    # Compare first 10 entries for equality (within tolerance)
    assert np.allclose(Y_student[:10], Y_expected[:10], atol=1e-6), (
        "endless_running_avg() does not correctly compute the running averages for the endless dungeon."
    )