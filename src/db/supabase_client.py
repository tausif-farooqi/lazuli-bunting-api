"""
Supabase client — single source of truth for all database access.
Import ``get_supabase_client`` wherever a Supabase connection is needed.
"""

from __future__ import annotations

from supabase import Client, create_client

from core.config import settings


def get_supabase_client() -> Client:
    if not settings.supabase_url or not settings.supabase_key:
        raise ValueError(
            "SUPABASE_URL and SUPABASE_SERVICE_KEY (or SUPABASE_ANON_KEY) "
            "must be set in the .env file."
        )
    return create_client(settings.supabase_url, settings.supabase_key)
