from otter.test_files import test_case

# This test verifies the convergence target values for each dungeon.

OK_FORMAT = False
name = "q8"
points = 1


@test_case(name="convergence targets", points=1)
def test_convergence_targets(convergence_targets):
    """Check that convergence_targets returns the correct limiting values for each dungeon."""
    result = convergence_targets()
    assert isinstance(result, dict), "convergence_targets() should return a dictionary."
    expected = {
        'Finite': 0.0,
        'Endless': 0.0,
        'Cursed': 0.0,
        'Greedy': 1.0,
    }
    assert result == expected, f"Expected {expected}, but got {result}."