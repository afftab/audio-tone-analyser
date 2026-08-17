"""Runtime configuration and secrets.

Secrets come from the environment (or a local `.env`, which is gitignored).
On a deployment they must come from the host's secret store -- HF Spaces
repository secrets, not the image.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Placeholders, not credentials -- they let the app start on a laptop with no
# .env. require_production_secrets() stops them reaching a deployment.
_DEV_USERNAME = "autoace"
_DEV_PASSWORD = "changeme"
_DEV_SESSION_SECRET = "dev-secret-change-in-production"

DASHBOARD_USERNAME = os.environ.get("DASHBOARD_USERNAME", _DEV_USERNAME)
DASHBOARD_PASSWORD = os.environ.get("DASHBOARD_PASSWORD", _DEV_PASSWORD)
SESSION_SECRET = os.environ.get("SESSION_SECRET", _DEV_SESSION_SECRET)

# Set by the Dockerfile. Anything other than "dev" is treated as a deployment.
ENVIRONMENT = os.environ.get("VTA_ENV", "dev").strip().lower()
IS_DEV = ENVIRONMENT == "dev"


class InsecureConfigError(RuntimeError):
    """A deployment was started while still using the placeholder credentials."""


def require_production_secrets() -> None:
    """Refuse to serve a deployment with the placeholders above.

    SESSION_SECRET is the session cookie's HMAC key, so a known value lets
    anyone mint an authenticated cookie and the login form stops being a
    control. The username and password are equally public -- they are in
    this file, in a repo that gets pushed to a public Space.
    """
    if IS_DEV:
        return
    insecure = [
        name
        for name, value, placeholder in (
            ("SESSION_SECRET", SESSION_SECRET, _DEV_SESSION_SECRET),
            ("DASHBOARD_PASSWORD", DASHBOARD_PASSWORD, _DEV_PASSWORD),
            ("DASHBOARD_USERNAME", DASHBOARD_USERNAME, _DEV_USERNAME),
        )
        if value == placeholder
    ]
    if insecure:
        raise InsecureConfigError(
            "Refusing to start with placeholder credentials in "
            f"VTA_ENV={ENVIRONMENT!r}: {', '.join(insecure)}. "
            "Set them as host secrets (HF Spaces: Settings -> Repository secrets). "
            "Generate a session secret with: python -c "
            "'import secrets; print(secrets.token_urlsafe(32))'"
        )
