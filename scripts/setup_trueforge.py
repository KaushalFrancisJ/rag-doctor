#!/usr/bin/env python3
"""
TrueForge Setup & Sync Script
-----------------------------
Automatically configures TrueForge runtime settings (Google Gemini Model Provider,
Daytona Sandbox Provider, and RAG Doctor MCP Server) from your .env and trueforge.yaml.

Usage:
  python scripts/setup_trueforge.py [--host http://localhost:8790]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
import urllib.error

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env(env_path: Path):
    """Load variables from .env into os.environ if not already set."""
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k not in os.environ:
                os.environ[k] = v


def make_request(url: str, method: str = "GET", data: dict | None = None) -> tuple[int, dict]:
    """Perform HTTP request to TrueForge API."""
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            res_data = json.loads(resp.read().decode("utf-8")) if resp.readable() else {}
            return status, res_data
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode("utf-8")
        try:
            res_data = json.loads(error_msg)
        except Exception:
            res_data = {"error": error_msg}
        return e.code, res_data
    except Exception as e:
        return 0, {"error": str(e)}


def wait_for_server(base_url: str, timeout_seconds: int = 15) -> bool:
    """Check if TrueForge server is online."""
    health_url = f"{base_url}/healthz"
    print(f"Connecting to TrueForge at {base_url}...")
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print("✅ TrueForge server is online.")
                    return True
        except Exception:
            time.sleep(1)
    return False


def configure_model_provider(base_url: str, gemini_api_key: str):
    """Configure native Google Gemini model provider."""
    print("\n[1/3] Configuring Google Gemini Model Provider...")
    if not gemini_api_key or gemini_api_key == "your_gemini_api_key_here":
        print("⚠️  GEMINI_API_KEY is not set or placeholder. Please update .env.")
        return False

    url = f"{base_url}/api/v1/settings/model-providers"
    payload = {
        "manifest": {
            "type": "google-gemini",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "auth": {
                "api_key": gemini_api_key
            },
            "models": [
                {
                    "model_id": "gemini-2.5-flash",
                    "name": "gemini-2-5-flash",
                    "properties": {
                        "context_length": 1048576,
                        "max_output_tokens": 65536,
                        "reasoning_efforts": ["minimal", "low", "medium", "high"]
                    }
                },
                {
                    "model_id": "gemini-2.5-pro",
                    "name": "gemini-2-5-pro",
                    "properties": {
                        "context_length": 1048576,
                        "max_output_tokens": 65536,
                        "reasoning_efforts": ["low", "medium", "high"]
                    }
                },
                {
                    "model_id": "gemini-2.0-flash",
                    "name": "gemini-2-0-flash",
                    "properties": {
                        "context_length": 1048576,
                        "max_output_tokens": 65536,
                        "reasoning_efforts": ["none", "low", "medium", "high"]
                    }
                }
            ]
        }
    }

    status, resp = make_request(url, method="PUT", data=payload)
    if status in (200, 201):
        print("✅ Google Gemini provider configured successfully.")
        return True
    else:
        print(f"❌ Failed to configure Gemini provider ({status}): {resp}")
        return False


def configure_sandbox_provider(base_url: str, daytona_api_key: str):
    """Configure Daytona Sandbox provider."""
    print("\n[2/3] Configuring Daytona Sandbox Provider...")
    if not daytona_api_key or daytona_api_key == "your_daytona_api_key_here":
        print("⚠️  DAYTONA_API_KEY is not set or placeholder. Please update .env.")
        return False

    url = f"{base_url}/api/v1/settings/sandbox-providers"
    payload = {
        "manifest": {
            "type": "daytona",
            "auth": {
                "api_key": daytona_api_key
            },
            "exec_timeout_ms": 60000,
            "auto_stop_interval_in_minutes": 5,
            "auto_archive_interval_in_minutes": 60,
            "auto_delete_interval_in_minutes": 7200
        }
    }

    status, resp = make_request(url, method="PUT", data=payload)
    if status in (200, 201):
        print("✅ Daytona Sandbox provider configured successfully.")
        return True
    else:
        print(f"❌ Failed to configure Daytona provider ({status}): {resp}")
        print("💡 Hint: Daytona requires a valid API key from https://app.daytona.io. Check DAYTONA_API_KEY in .env.")
        return False


def configure_mcp_server(base_url: str, mcp_url: str = "http://127.0.0.1:8000/sse"):
    """Register RAG Doctor MCP server."""
    print("\n[3/3] Registering RAG Doctor MCP Server...")
    url = f"{base_url}/api/v1/settings/mcp-servers"
    payload = {
        "manifest": {
            "type": "remote",
            "name": "rag-doctor-mcp",
            "url": mcp_url,
            "description": "RAG Doctor MCP server for inspecting pipeline health and diagnostics"
        }
    }

    status, resp = make_request(url, method="PUT", data=payload)
    if status in (200, 201):
        print("✅ RAG Doctor MCP server registered successfully.")
        return True
    else:
        print(f"❌ Failed to register MCP server ({status}): {resp}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Configure TrueForge with Google Gemini and Daytona Sandbox.")
    parser.add_argument("--host", default=os.getenv("TRUEFORGE_HOST", "http://localhost:8790"), help="TrueForge base URL")
    parser.add_argument("--mcp-url", default=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/sse"), help="RAG Doctor MCP server SSE endpoint URL")
    parser.add_argument("--wait-timeout", type=int, default=30, help="Seconds to wait for TrueForge server to become available")
    args = parser.parse_args()

    env_path = PROJECT_ROOT / ".env"
    load_env(env_path)

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENERATIVE_AI_API_KEY", "")
    daytona_key = os.getenv("DAYTONA_API_KEY", "")
    mcp_url = args.mcp_url

    if not wait_for_server(args.host, timeout_seconds=args.wait_timeout):
        print(f"\n⚠️  TrueForge is not running on {args.host}.")
        print("To start TrueForge, run in another terminal:")
        print("  npx @truefoundry/trueforge")
        print("\nOnce TrueForge starts, re-run:")
        print("  python scripts/setup_trueforge.py")
        sys.exit(1)

    configure_model_provider(args.host, gemini_key)
    configure_sandbox_provider(args.host, daytona_key)
    configure_mcp_server(args.host, mcp_url)

    print("\n🎉 TrueForge configuration complete!")


if __name__ == "__main__":
    main()
