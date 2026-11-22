from commonplace.config import Settings


def test_settings_defaults_point_at_local_stack() -> None:
    s = Settings(_env_file=None)
    assert s.database_url.endswith("@localhost:5433/commonplace")
    assert s.inngest_dev is True
