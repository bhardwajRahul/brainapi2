from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from deploy.backup_restore import verify_backup


ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"


@pytest.mark.parametrize("profile", ["light", "heavy"])
def test_only_nginx_publishes_ports(profile):
    data = yaml.safe_load(
        (DEPLOY / f"docker-compose.{profile}.yaml").read_text(encoding="utf-8")
    )
    published = {
        service: config["ports"]
        for service, config in data["services"].items()
        if config.get("ports")
    }
    assert published == {"nginx": ["80:80", "443:443"]}
    assert "version" not in data


def _compose_env(profile: str) -> dict[str, str]:
    env = {
        **os.environ,
        "BRAINAPI_IMAGE": "brainapi:test",
        "BRAINAPI_ENV_FILE": str(DEPLOY / f"env.{profile}.example"),
        "REDIS_PASSWORD": "redis-secret",
    }
    if profile == "light":
        env["POSTGRES_PASSWORD"] = "postgres-secret"
    else:
        env.update(
            {
                "NEO4J_PASSWORD": "neo-secret",
                "MONGO_PASSWORD": "mongo-secret",
                "MINIO_ACCESS_KEY": "minio-access",
                "MINIO_SECRET_KEY": "minio-secret",
                "MILVUS_TOKEN": "root:milvus-secret",
                "MILVUS_ROOT_PASSWORD": "milvus-secret",
            }
        )
    return env


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker unavailable")
@pytest.mark.parametrize("profile", ["light", "heavy"])
def test_compose_profiles_render_with_escaped_healthchecks(profile):
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(DEPLOY / f"docker-compose.{profile}.yaml"),
            "config",
            "--format",
            "json",
        ],
        env=_compose_env(profile),
        text=True,
        capture_output=True,
        check=True,
    )
    config = json.loads(result.stdout)
    assert config["services"]["redis"]["healthcheck"]["test"][1].find(
        "REDIS_PASSWORD"
    ) >= 0
    assert set(config["services"]["nginx"]["ports"][0]) >= {"target", "published"}


@pytest.mark.skipif(shutil.which("docker") is None, reason="Docker unavailable")
def test_compose_rejects_missing_required_secret():
    env = _compose_env("light")
    env.pop("REDIS_PASSWORD")
    result = subprocess.run(
        ["docker", "compose", "-f", str(DEPLOY / "docker-compose.light.yaml"), "config", "-q"],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "REDIS_PASSWORD is required" in result.stderr


def test_nginx_routes_console_api_and_mcp():
    config = (DEPLOY / "nginx" / "conf.d" / "brainapi.conf").read_text(
        encoding="utf-8"
    )
    assert "location /mcp" in config
    assert "location /" in config
    assert "brainapi_api" in config
    assert "brainapi_mcp" in config


def test_backup_manifest_verification_detects_tampering(tmp_path):
    artifact = tmp_path / "postgres.dump"
    artifact.write_bytes(b"consistent")
    manifest = {
        "format_version": 1,
        "profile": "light",
        "checksums": {
            artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    verify_backup(tmp_path, "light")
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify_backup(tmp_path, "light")


def test_recommendation_openapi_surface_is_preview():
    from src.services.api.app import app

    schema = app.openapi()
    assert schema["paths"]["/retrieve/recommend"]["get"]["x-stability"] == "preview"
    assert schema["paths"]["/retrieve/recommend"]["post"]["x-stability"] == "preview"
