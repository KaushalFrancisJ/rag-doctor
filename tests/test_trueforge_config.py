"""
Unit tests for TrueForge configuration files and setup script.
"""

from pathlib import Path
import yaml


def test_trueforge_yaml_valid():
    config_path = Path("trueforge.yaml")
    assert config_path.exists(), "trueforge.yaml should exist"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    assert "models" in data
    assert data["models"]["provider"]["type"] == "google-gemini"
    assert "sandbox" in data
    assert data["sandbox"]["provider"]["type"] == "daytona"
    assert "agent" in data
    assert data["agent"]["config"]["sandbox"]["enabled"] is True


def test_trueforge_catalogs():
    models_catalog = Path("trueforge-models.yaml")
    sandbox_catalog = Path("trueforge-sandbox.yaml")
    mcp_catalog = Path("trueforge-mcp.yaml")

    assert models_catalog.exists()
    assert sandbox_catalog.exists()
    assert mcp_catalog.exists()

    with open(models_catalog, "r", encoding="utf-8") as f:
        m_data = yaml.safe_load(f)
        assert "providers" in m_data
        assert any(p["type"] == "google-gemini" for p in m_data["providers"])

    with open(sandbox_catalog, "r", encoding="utf-8") as f:
        s_data = yaml.safe_load(f)
        assert "providers" in s_data
        assert any(p["type"] == "daytona" for p in s_data["providers"])

    with open(mcp_catalog, "r", encoding="utf-8") as f:
        mcp_data = yaml.safe_load(f)
        assert "mcp_servers" in mcp_data


def test_docker_compose_valid():
    compose_path = Path("docker-compose.yml")
    assert compose_path.exists()
    with open(compose_path, "r", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)
    assert "services" in compose_data
    assert "mcp-server" in compose_data["services"]
    assert "trueforge" in compose_data["services"]
    assert "init-config" in compose_data["services"]
