from otter.test_files import test_case

# This test file checks that the student has implemented a function
# called ``finite_final_loot`` which returns the final treasure count for
# the finite dungeon. We call the function and verify its return value.

OK_FORMAT = False
name = "q2"
points = 1

@test_case(name="finite final loot", points=1)
def test_finite_final(finite_final_loot):
    """
    Check that finite_final_loot returns the correct final treasure with a fixed seed.

    The expected result is computed by directly simulating the finite dungeon (p_n = n^{-1.3})
    using the same random seed.
    """
    import numpy as np
    # Reset RNG for student function
    np.random.seed(42)
    student_result = finite_final_loot()
    # Manual simulation
    np.random.seed(42)
    X = np.zeros(1000, dtype=float)
    for n in range(1, 1001):
        p = n ** (-1.3)
        if np.random.rand() < p:
            X[n - 1] = 1.0
    expected_result = float(np.sum(X))
    assert int(student_result) == int(expected_result), (
        f"The final loot in the finite dungeon (N=1000, seed=42) should be {int(expected_result)}, but got {student_result}."
    )
