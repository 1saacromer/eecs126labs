from otter.test_files import test_case

# This test file checks that the student has implemented a function
# called ``greedy_loot`` which returns a tuple containing the final
# treasure and the average treasure per room in the greedy dungeon. The
# greedy dungeon has N=1000 rooms and uses a seed of 42 for
# reproducibility. The correct final treasure and average were
# precomputed with this seed.

import numpy as np

OK_FORMAT = False
name = "q5"
points = 3

@test_case(name="greedy final values", points=3)
def test_greedy_final(greedy_loot):
    """
    Check that greedy_loot returns the correct total and average treasure with a fixed seed.

    The expected total is recomputed via a direct simulation of the greedy dungeon with
    the same random seed, rather than relying on a hard-coded constant.
    """
    import numpy as np
    # Reset RNG for student's function
    np.random.seed(42)
    result = greedy_loot()
    assert isinstance(result, tuple) and len(result) == 2, "greedy_loot() should return a tuple of length 2."
    total, avg = result
    # Directly simulate the greedy dungeon to compute the expected total
    np.random.seed(42)
    manual_total = 0.0
    for n in range(1, 1001):
        if np.random.rand() < 1.0 / n:
            manual_total += n
    # Compare the student's total to the manual total
    assert abs(float(total) - manual_total) < 1e-6, (
        f"The final treasure in the greedy dungeon should be {manual_total}, but got {total}."
    )
    # The average should equal total divided by 1000
    assert abs(avg - total / 1000) < 1e-6, "The average treasure should equal the total divided by 1000."