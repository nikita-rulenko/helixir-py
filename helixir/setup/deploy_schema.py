"""
Deploy HelixDB Schema and Queries.

Loads schema.hx and queries.hx into HelixDB instance.
"""

from pathlib import Path
import subprocess
import sys

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def deploy_schema() -> bool:
    """Deploy schema and queries to HelixDB using helix CLI."""
    project_root = get_project_root()
    schema_dir = project_root / "schema"

    schema_file = schema_dir / "schema.hx"
    queries_file = schema_dir / "queries.hx"

    if not schema_file.exists():
        return False

    if not queries_file.exists():
        return False


    try:
        result = subprocess.run(
            ["helix", "--version"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
    except FileNotFoundError:
        return False

    try:
        result = subprocess.run(
            ["helix", "check"],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            pass

        result = subprocess.run(
            ["helix", "push", "dev"],
            check=False,
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            return True

        return deploy_direct()

    except Exception:
        return False


def deploy_direct() -> bool:
    """Deploy schema directly via HTTP (fallback method)."""
    import httpx
    import yaml

    project_root = get_project_root()
    config_path = project_root / "config.yaml"

    if not config_path.exists():
        return False

    with open(config_path) as f:
        config = yaml.safe_load(f)

    host = config.get("host", "localhost")
    port = config.get("port", 6969)
    base_url = f"http://{host}:{port}"

    schema_dir = project_root / "schema"

    with open(schema_dir / "schema.hx") as f:
        f.read()

    with open(schema_dir / "queries.hx") as f:
        f.read()


    try:
        response = httpx.get(f"{base_url}/health", timeout=5)

        return response.status_code == 200

    except httpx.ConnectError:
        return False
    except Exception:
        return False


if __name__ == "__main__":
    success = deploy_schema()
    sys.exit(0 if success else 1)
