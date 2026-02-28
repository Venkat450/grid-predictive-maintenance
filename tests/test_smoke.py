from src.config import ProjectConfig


def test_project_config_defaults():
    cfg = ProjectConfig()
    assert cfg.model_path.endswith("xgboost_model.pkl")
    assert 0 < cfg.test_size < 1
