"""
Interactive Configuration Wizard for Helixir.

Guides user through setting up config.yaml and mcp.json.
"""

import json
from pathlib import Path

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a section header."""


def print_success(text: str) -> None:
    """Print success message."""


def print_warning(text: str) -> None:
    """Print warning message."""


def print_error(text: str) -> None:
    """Print error message."""


def prompt(question: str, default: str = "") -> str:
    """Prompt user for input with optional default."""
    if default:
        user_input = input(f"{question} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{question}: ").strip()


def prompt_choice(question: str, choices: list[str], default: str = "") -> str:
    """Prompt user to choose from options."""
    choices_str = "/".join(choices)
    if default:
        return prompt(f"{question} ({choices_str})", default)
    return prompt(f"{question} ({choices_str})")


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def generate_config(settings: dict) -> str:
    """Generate config.yaml content from settings."""
    return f"""# ============================================================

llm_provider: "{settings["llm_provider"]}"
llm_model: "{settings["llm_model"]}"
llm_temperature: 0.3
llm_api_key: {f'"{settings["llm_api_key"]}"' if settings.get("llm_api_key") else "null"}
llm_base_url: {f'"{settings["llm_base_url"]}"' if settings.get("llm_base_url") else "null"}

embedding_provider: "{settings["embedding_provider"]}"
embedding_model: "{settings["embedding_model"]}"
embedding_url: "{settings["embedding_url"]}"
embedding_api_key: {f'"{settings["embedding_api_key"]}"' if settings.get("embedding_api_key") else "null"}

host: "{settings["helix_host"]}"
port: {settings["helix_port"]}
instance: "{settings["helix_instance"]}"
timeout: 30
"""


def generate_mcp_config(settings: dict, project_path: Path) -> dict:
    """Generate MCP server configuration."""
    env = {
        "HELIX_HOST": settings["helix_host"],
        "HELIX_PORT": str(settings["helix_port"]),
        "HELIX_INSTANCE": settings["helix_instance"],
        "HELIX_LLM_PROVIDER": settings["llm_provider"],
        "HELIX_LLM_MODEL": settings["llm_model"],
    }

    if settings.get("llm_api_key"):
        env["HELIX_LLM_API_KEY"] = settings["llm_api_key"]

    if settings.get("llm_base_url"):
        env["HELIX_LLM_BASE_URL"] = settings["llm_base_url"]

    return {
        "command": str(project_path / ".venv" / "bin" / "python"),
        "args": ["-m", "helixir.mcp.server"],
        "cwd": str(project_path),
        "env": env,
    }


def patch_mcp_json(mcp_config: dict) -> bool:
    """Add Helixir to ~/.cursor/mcp.json."""
    mcp_path = Path.home() / ".cursor" / "mcp.json"

    if not mcp_path.parent.exists():
        print_warning("Cursor config directory not found (~/.cursor/)")
        return False

    if mcp_path.exists():
        with open(mcp_path) as f:
            config = json.load(f)
    else:
        config = {"mcpServers": {}}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    config["mcpServers"]["helixir"] = mcp_config

    with open(mcp_path, "w") as f:
        json.dump(config, f, indent=2)

    return True


def run_wizard() -> None:
    """Run the interactive setup wizard."""
    print_header("Helixir Configuration Wizard")

    project_root = get_project_root()
    settings = {}

    settings["helix_host"] = prompt("  Host", "localhost")
    settings["helix_port"] = int(prompt("  Port", "6969"))
    settings["helix_instance"] = prompt("  Instance name", "dev")


    settings["llm_provider"] = prompt_choice(
        "  Provider", ["cerebras", "ollama", "openai"], "cerebras"
    )

    if settings["llm_provider"] == "cerebras":
        settings["llm_model"] = prompt("  Model", "llama3.3-70b")
        settings["llm_api_key"] = prompt("  API Key (get free at https://cloud.cerebras.ai)")
        settings["llm_base_url"] = None
    elif settings["llm_provider"] == "ollama":
        settings["llm_model"] = prompt("  Model", "llama3.2")
        settings["llm_base_url"] = prompt("  Ollama URL", "http://localhost:11434")
        settings["llm_api_key"] = None
    else:
        settings["llm_model"] = prompt("  Model", "gpt-4o")
        settings["llm_api_key"] = prompt("  API Key")
        settings["llm_base_url"] = None


    settings["embedding_provider"] = prompt_choice("  Provider", ["ollama", "openai"], "ollama")

    if settings["embedding_provider"] == "ollama":
        settings["embedding_model"] = prompt("  Model", "nomic-embed-text")
        settings["embedding_url"] = prompt("  Ollama URL", "http://localhost:11434")
        settings["embedding_api_key"] = None
    else:
        settings["embedding_model"] = prompt("  Model", "text-embedding-3-small")
        settings["embedding_url"] = "https://api.openai.com/v1"
        settings["embedding_api_key"] = prompt("  API Key")

    print_header("Saving Configuration")

    config_content = generate_config(settings)
    config_path = project_root / "config.yaml"

    with open(config_path, "w") as f:
        f.write(config_content)

    print_success(f"Config saved to {config_path}")

    add_mcp = prompt("  Add Helixir to Cursor MCP? (y/n)", "y").lower()

    if add_mcp == "y":
        mcp_config = generate_mcp_config(settings, project_root)
        if patch_mcp_json(mcp_config):
            print_success("Added to ~/.cursor/mcp.json")
            print_warning("Restart Cursor to activate MCP server")
        else:
            print_warning("Could not patch mcp.json - add manually")

    print_header("Configuration Complete!")


if __name__ == "__main__":
    run_wizard()
