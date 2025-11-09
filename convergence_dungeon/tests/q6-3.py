from otter.test_files import test_case
import numpy as np

# This test verifies the running averages for the greedy dungeon.

OK_FORMAT = False
name = "q6-3"
points = 4


@test_case(name="greedy running average reproducibility", points=1)
def test_greedy_running_avg_reproducibility(greedy_running_avg):
    """greedy_running_avg should produce the same result when the global RNG seed is reset."""
    import numpy as np
    np.random.seed(42)
    Y1 = greedy_running_avg()
    np.random.seed(42)
    Y2 = greedy_running_avg()
    assert isinstance(Y1, np.ndarray) and isinstance(Y2, np.ndarray), "greedy_running_avg() should return a numpy array."
    assert np.array_equal(Y1, Y2), "greedy_running_avg() should be reproducible given a fixed seed."


@test_case(name="greedy running average shape and final value", points=3)
def test_greedy_running_avg_shape_and_final(greedy_running_avg):
    """
    Check the length and final value of the running average for the greedy dungeon.

    The expected final running average is recomputed via a direct simulation of the greedy
    dungeon under the same seed. This avoids relying on a hard-coded value.
    """
    import numpy as np
    # Reset the RNG for the student's function
    np.random.seed(42)
    Y_student = greedy_running_avg()
    assert isinstance(Y_student, np.ndarray), "greedy_running_avg() should return a numpy array."
    assert Y_student.shape == (1000,), "The running average array must have length 1000."
    # Direct simulation of the greedy dungeon
    np.random.seed(42)
    manual_X = np.zeros(1000, dtype=float)
    for n in range(1, 1001):
        if np.random.rand() < 1.0 / n:
            manual_X[n - 1] = float(n)
    manual_S = np.cumsum(manual_X)
    manual_Y = manual_S / np.arange(1, 1001)
    # Compare the final running average
    assert abs(float(Y_student[-1]) - float(manual_Y[-1])) < 1e-6, (
        f"The final running average in the greedy dungeon should be {manual_Y[-1]:.6f}."
    )