from otter.test_files import test_case
import numpy as np

OK_FORMAT = False
name = "q1"
points = 4

# These tests verify that the simulate_dungeon helper function behaves correctly.

@test_case(name="simulate_dungeon output format", points=2)
def test_simulate_shape(simulate_dungeon):
    """
    simulate_dungeon should return two numpy arrays X and S of shape (N,).
    X contains 0/1 indicators and S is the cumulative sum of X.
    """
    import numpy as np
    np.random.seed(0)
    # Use a constant probability p_n = 0.5 for a small number of rooms
    X, S = simulate_dungeon(lambda n: 0.5, 10)
    assert isinstance(X, np.ndarray), "X should be a numpy array."
    assert isinstance(S, np.ndarray), "S should be a numpy array."
    assert X.shape == (10,), "X should have shape (N,) when N=10."
    assert S.shape == (10,), "S should have shape (N,) when N=10."
    # Check that S is the cumulative sum of X
    assert np.allclose(S, np.cumsum(X)), "S must be the cumulative sum of X."


@test_case(name="simulate_dungeon final value", points=2)
def test_simulate_values(simulate_dungeon):
    """
    For a fair coin (p=0.5) and N=10, the final loot count is deterministic given a seed.
    """
    import numpy as np
    np.random.seed(0)
    X, S = simulate_dungeon(lambda n: 0.5, 10)
    # With seed=0, p=0.5, the expected final loot count is 3
    assert int(S[-1]) == 3, "For p=0.5 and N=10 with seed=0, the final loot should be 3."