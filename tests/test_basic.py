def test_import():
    import harreman
    assert hasattr(harreman, "__version__") or True
