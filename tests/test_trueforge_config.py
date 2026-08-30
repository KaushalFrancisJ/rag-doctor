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
    assert data["agent"]["name"] == "rag-doctor"
    assert data["agent"]["config"]["sandbox"]["enabled"] is True
    assert data["agent"]["config"]["dynamic_sub_agents"]["enabled"] is True

    # Verify subagents
    assert "subagents" in data
    diagnose_sub = next((s for s in data["subagents"] if s["name"] == "diagnose"), None)
    assert diagnose_sub is not None, "Diagnose subagent must be defined in trueforge.yaml"
    assert diagnose_sub["config"]["sandbox"]["enabled"] is False  # Read-only
    assert diagnose_sub["config"]["dynamic_sub_agents"]["enabled"] is False
    assert diagnose_sub["response_format"]["type"] == "json_schema"

    schema = diagnose_sub["response_format"]["json_schema"]["schema"]
    required_fields = set(schema["required"])
    assert {"suspected_cause", "evidence", "confidence", "hypothesis", "recommended_experiment"} <= required_fields
    assert len(schema["properties"]["suspected_cause"]["enum"]) == 7

    # Verify Fix subagent
    fix_sub = next((s for s in data["subagents"] if s["name"] == "fix"), None)
    assert fix_sub is not None, "Fix subagent must be defined in trueforge.yaml"
    assert fix_sub["config"]["sandbox"]["enabled"] is False
    assert fix_sub["config"]["dynamic_sub_agents"]["enabled"] is False
    assert fix_sub["response_format"]["type"] == "json_schema"

    fix_schema = fix_sub["response_format"]["json_schema"]["schema"]
    assert {"hypothesis", "strategy", "changes", "expected_effect", "reasoning"} <= set(fix_schema["required"])
    assert fix_schema["properties"]["strategy"]["enum"] == ["chunking", "retrieval"]
    changes_schema = fix_schema["properties"]["changes"]
    assert changes_schema.get("minProperties") == 1
    assert changes_schema["properties"]["chunk_size"]["minimum"] == 50
    assert changes_schema["properties"]["top_k"]["minimum"] == 1


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
        
        # Verify openai-compatible provider for nvidia-nim-litellm
        nim_provider = next((p for p in m_data["providers"] if p.get("type") == "openai"), None)
        assert nim_provider is not None, "openai provider for nvidia-nim-litellm must be in trueforge-models.yaml"
        
        nim_models = nim_provider.get("models", [])
        assert any(m["model_id"] == "nvidia-nemotron-lightning" for m in nim_models)
        nemotron = next(m for m in nim_models if m["model_id"] == "nvidia-nemotron-lightning")
        assert nemotron["name"] == "nvidia-nemotron-lightning"
        assert nemotron["properties"]["context_length"] == 1000000

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

    # Ensure ports are bound to loopback 127.0.0.1 for security
    mcp_ports = compose_data["services"]["mcp-server"].get("ports", [])
    assert any("127.0.0.1:8000" in p for p in mcp_ports)

    tf_ports = compose_data["services"]["trueforge"].get("ports", [])
    assert any("127.0.0.1:8790" in p for p in tf_ports)

    # Ensure persistent trueforge-data volume is mounted to preserve chats and database
    assert "volumes" in compose_data
    assert "trueforge-data" in compose_data["volumes"]
    tf_volumes = compose_data["services"]["trueforge"].get("volumes", [])
    assert any("trueforge-data:/root/.trueforge" in str(v) for v in tf_volumes)


def test_setup_script_model_loader(monkeypatch, tmp_path):
    from scripts.setup_trueforge import load_models_from_catalog

    # 1. Default catalog loading
    models = load_models_from_catalog()
    assert len(models) >= 3
    model_ids = [m["model_id"] for m in models]
    assert "gemini-2.5-flash" in model_ids
    assert "gemini-2.5-pro" in model_ids
    assert "gemini-2.0-flash" in model_ids

    # 2. Loading via explicit relative path
    models_rel = load_models_from_catalog("./trueforge-models.yaml")
    assert len(models_rel) == len(models)

    # 3. Loading via MODEL_CATALOG_PATH environment variable
    monkeypatch.setenv("MODEL_CATALOG_PATH", "./trueforge-models.yaml")
    models_env = load_models_from_catalog()
    assert len(models_env) == len(models)

    # 4. Loading from custom catalog path
    custom_catalog = tmp_path / "custom-models.yaml"
    custom_catalog.write_text(
        yaml.dump({
            "providers": [
                {
                    "type": "google-gemini",
                    "models": [
                        {"model_id": "custom-gemini-model", "name": "custom-model"}
                    ]
                }
            ]
        }),
        encoding="utf-8"
    )
    models_custom = load_models_from_catalog(str(custom_catalog))
    assert len(models_custom) == 1
    assert models_custom[0]["model_id"] == "custom-gemini-model"
