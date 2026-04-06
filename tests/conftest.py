import pytest

def pytest_addoption(parser):
    parser.addoption("--snax-sim", action="store_true", default=False,
                     help="Run tests that require Verilator SNAX simulation")
    parser.addoption("--tolerance", type=float, default=1e-3,
                     help="Numerical tolerance for embedded vs reference comparison")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--snax-sim"):
        skip_snax = pytest.mark.skip(reason="needs --snax-sim flag")
        for item in items:
            if "snax" in item.keywords:
                item.add_marker(skip_snax)
