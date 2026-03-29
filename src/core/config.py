"""
Centralised application settings loaded from environment / .env file.
All other modules should import from here instead of reading os.environ directly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: src/core/config.py  →  ../../  (lazuli-bunting-api/)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
MODELS_DIR: Path = ROOT_DIR / "models"


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    ebird_api_key: str = ""
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def supabase_key(self) -> str:
        return self.supabase_service_key or self.supabase_anon_key


settings = Settings()
