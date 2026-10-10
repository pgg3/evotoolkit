import evotoolkit


def test_public_api():
    for name in evotoolkit.__all__:
        assert hasattr(evotoolkit, name)
    assert evotoolkit.__version__.startswith("2.")
    assert not hasattr(evotoolkit, "EoH")
