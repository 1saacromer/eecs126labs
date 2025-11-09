from otter.test_files import test_case

# This test verifies that the classify_convergence function returns the correct modes for each dungeon.

OK_FORMAT = False
name = "q7"
points = 3

@test_case(name="modes of convergence classification", points=2)
def test_classify_convergence(classify_convergence):
    """Check that the convergence classification dictionary is correct."""
    result = classify_convergence()
    assert isinstance(result, dict), "The result should be a dictionary."
    expected_keys = {"Finite", "Endless", "Cursed", "Greedy"}
    assert set(result.keys()) == expected_keys, "Dictionary keys must be exactly Finite, Endless, Cursed, Greedy."
    assert set(result["Finite"]) == {"almost surely", "in distribution"}, "Finite dungeon should converge almost surely and in distribution."
    assert set(result["Endless"]) == {"in probability"}, "Endless dungeon should converge only in probability."
    assert result["Cursed"] == [], "Cursed dungeon has no guaranteed convergence modes."
    assert set(result["Greedy"]) == {"in probability"}, "Greedy dungeon should converge in probability but not in distribution."


@test_case(name="no 'almost sure' string used", points=1)
def test_no_almost_sure_string(classify_convergence):
    """Ensure the function does not mistakenly return 'almost sure' instead of 'almost surely'."""
    result = classify_convergence()
    for modes in result.values():
        assert "almost sure" not in modes, "Use 'almost surely' (adverb) rather than 'almost sure' in convergence modes."
