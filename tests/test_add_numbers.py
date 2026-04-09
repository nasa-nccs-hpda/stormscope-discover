def add_numbers(a, b):
    return a + b


def test_add_numbers():
    assert add_numbers(2, 3) == 5  # Test positive integers
    assert add_numbers(-1, 1) == 0  # Test negative and positive
    assert add_numbers(0, 0) == 0  # Test zero
    assert add_numbers(2.5, 3.5) == 6.0  # Test floats


# Run tests if the script is executed directly
if __name__ == "__main__":

    import pytest
    pytest.main([__file__])
