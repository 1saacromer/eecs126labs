from otter.test_files import test_case

OK_FORMAT = False
name = "q4"
points = 1

@test_case(name="cursed final loot", points=1)
def test_cursed_final(cursed_final_loot):
    """
    Check that the cursed dungeon returns the correct final treasure count with a fixed seed.

    Expected result is computed by simulating the same dependent process:
    base p_n = 1/n; after a success multiply by 2 (cap at 1); after a failure multiply by 0.5.
    We reseed before both runs so the student code and manual simulation use identical draws.
    """
    import numpy as np

    # Student result (student code must NOT reseed internally)
    np.random.seed(42)
    student_result = cursed_final_loot()

    # Manual simulation for expected result
    np.random.seed(42)
    total = 0.0
    state = 0  # 0 = unlucky (halve), 1 = lucky (double)
    for n in range(1, 1001):
        p_base = 1.0 / n
        p = min(2.0 * p_base, 1.0) if state == 1 else 0.5 * p_base
        if np.random.rand() < p:
            total += 1.0
            state = 1
        else:
            state = 0
    expected_result = float(total)

    # Compare as integers (exact match under the same RNG stream)
    assert int(student_result) == int(expected_result), (
        f"The final loot in the cursed dungeon (N=1000, seed=42) should be {int(expected_result)}, "
        f"but got {student_result}."
    )