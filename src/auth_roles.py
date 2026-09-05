"""
AuraPredict — Role-Based Authorization  (Fase 7)
=================================================
Centralised role checks for all API endpoints.

Role hierarchy (ascending privilege):
  VIEWER      → read-only: health, historial, anomalías, reporting
  MAINTENANCE → VIEWER + write ground-truth, trigger training, alerts
  ENGINEER    → MAINTENANCE + model management (activate, rollback)
  ADMIN       → full access: manage machines, plants, lines, users, exports

Design:
  - Roles come from the JWT payload field "rol".
  - All existing legacy roles ("admin", "usuario") are mapped.
  - FastAPI Depends() injection pattern — no modification to existing routes needed.
  - Backward-compatible: endpoints with no role check keep working.

Usage in api.py:
    from auth_roles import require_roles, ROLE_ADMIN, ROLE_ENGINEER

    @app.post("/v2/maquinas/{id}/modelos/{mid}/activar")
    def activar_modelo(
        maquina_id: int, model_id: int,
        current_user: dict = Depends(require_roles(ROLE_ENGINEER, ROLE_ADMIN)),
    ):
        ...
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

# fastapi imported lazily to allow importing auth_roles without fastapi installed


# ── Role constants ─────────────────────────────────────────────────────────────

ROLE_VIEWER      = "viewer"
ROLE_MAINTENANCE = "maintenance"
ROLE_ENGINEER    = "engineer"
ROLE_ADMIN       = "admin"

# Canonical role set — what each level can do
ROLE_HIERARCHY: dict[str, int] = {
    ROLE_VIEWER:      1,
    ROLE_MAINTENANCE: 2,
    ROLE_ENGINEER:    3,
    ROLE_ADMIN:       4,
    # Legacy role aliases
    "usuario":        2,   # legacy "usuario" → maintenance level
    "tecnico":        2,
}

# Operations allowed per role level
PERMISSIONS: dict[str, set[str]] = {
    ROLE_VIEWER: {
        "view_health", "view_historial", "view_anomalias", "view_modelos",
        "view_reporting", "view_resumen",
    },
    ROLE_MAINTENANCE: {
        "view_health", "view_historial", "view_anomalias", "view_modelos",
        "view_reporting", "view_resumen",
        "write_ground_truth", "export_data",
    },
    ROLE_ENGINEER: {
        "view_health", "view_historial", "view_anomalias", "view_modelos",
        "view_reporting", "view_resumen",
        "write_ground_truth", "export_data",
        "activate_model", "rollback_model", "train_model",
    },
    ROLE_ADMIN: {
        "view_health", "view_historial", "view_anomalias", "view_modelos",
        "view_reporting", "view_resumen",
        "write_ground_truth", "export_data",
        "activate_model", "rollback_model", "train_model",
        "manage_machines", "manage_plants", "manage_users",
    },
}


# ── FastAPI dependency factories ───────────────────────────────────────────────

def require_roles(*allowed_roles: str) -> Callable:
    """
    FastAPI Depends() factory that restricts endpoint access to specific roles.

    Usage:
        @app.post("/v2/.../activar")
        def activate(
            current_user: dict = Depends(require_roles(ROLE_ENGINEER, ROLE_ADMIN))
        ):
            ...

    Returns a dependency that:
      1. Validates JWT via the existing get_usuario_actual dependency.
      2. Checks that user's rol is in allowed_roles.
      3. Raises HTTP 403 if not.
    """
    from api import get_usuario_actual   # lazy to avoid circular import
    from fastapi import Depends, HTTPException

    async def _check_roles(current_user: dict = Depends(get_usuario_actual)):
        user_role = current_user.get("rol", "")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Acceso denegado. Rol requerido: {', '.join(allowed_roles)}. "
                    f"Tu rol: {user_role or 'sin rol'}"
                ),
            )
        return current_user

    return _check_roles


def require_permission(permission: str) -> Callable:
    """
    FastAPI Depends() factory that checks a specific permission string.
    More granular than require_roles when a permission spans multiple roles.
    """
    from api import get_usuario_actual
    from fastapi import Depends, HTTPException

    async def _check_permission(current_user: dict = Depends(get_usuario_actual)):
        user_role = current_user.get("rol", "")
        user_level = ROLE_HIERARCHY.get(user_role, 0)

        # Find all roles whose PERMISSIONS include this permission
        allowed = {
            role for role, perms in PERMISSIONS.items()
            if permission in perms
        }
        allowed_levels = {ROLE_HIERARCHY.get(r, 0) for r in allowed}
        min_required = min(allowed_levels) if allowed_levels else 99

        if user_level < min_required:
            raise HTTPException(
                status_code=403,
                detail=f"Permiso requerido: {permission}. Tu rol '{user_role}' no tiene acceso.",
            )
        return current_user

    return _check_permission


def get_user_role_level(current_user: dict) -> int:
    """Return numeric role level for a user dict. 0 = unknown."""
    return ROLE_HIERARCHY.get(current_user.get("rol", ""), 0)


def is_admin(current_user: dict) -> bool:
    return current_user.get("rol") == ROLE_ADMIN


def can_write(current_user: dict) -> bool:
    """True for maintenance, engineer, admin."""
    return get_user_role_level(current_user) >= ROLE_HIERARCHY[ROLE_MAINTENANCE]


def can_manage_models(current_user: dict) -> bool:
    """True for engineer and admin."""
    return get_user_role_level(current_user) >= ROLE_HIERARCHY[ROLE_ENGINEER]
