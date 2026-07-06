import pytest

from app import config as app_config


def pytest_configure(config):
    config.addinivalue_line("markers", "live: test needs the live Vertex AI API")


def pytest_collection_modifyitems(config, items):
    if app_config.credentials_present():
        return
    skip_live = pytest.mark.skip(reason="no GOOGLE_APPLICATION_CREDENTIALS; skipping live API test")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
