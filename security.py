"""Authentification par clé d'API, limitation de débit, génération de clés.

Les clés ne sont JAMAIS stockées en clair : seul leur hash SHA-256 est dans
l'environnement (FINPREDICT_API_KEY_HASHES="nom1:hash1,nom2:hash2").

Générer une clé :   python security.py gen-key analyste-1
Le script affiche la clé (à donner au client, une seule fois) et la ligne à
ajouter dans .env.

Le limiteur de débit est en mémoire (par processus). Avec plusieurs workers ou
instances, utilisez Redis ou le rate limiting de la passerelle (Azure API
Management / Front Door).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sys
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load_key_hashes(raw: str | None) -> dict[str, str]:
    keys: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, digest = item.partition(":")
        if not sep or len(digest) != 64:
            raise ValueError("FINPREDICT_API_KEY_HASHES : format attendu 'nom:sha256hex' (64 caractères).")
        keys[name.strip()] = digest.strip().lower()
    return keys


def find_key_name(presented: str, key_hashes: dict[str, str]) -> str | None:
    presented_hash = hash_key(presented)
    match = None
    for name, stored in key_hashes.items():  # pas de sortie anticipée : temps constant
        if hmac.compare_digest(presented_hash, stored):
            match = name
    return match


class RateLimiter:
    """Fenêtre glissante de 60 s, par identité."""

    def __init__(self, window_seconds: int = 60):
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str, limit: int, *, record: bool = True) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            q = self._hits[identity]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= limit:
                return False, max(1, int(self.window - (now - q[0])) + 1)
            if record:
                q.append(now)
            return True, 0


class ApiAuth:
    """Dépendance FastAPI : clé d'API + rate limiting. Échoue en mode fermé."""

    def __init__(self):
        self.key_hashes = load_key_hashes(os.getenv("FINPREDICT_API_KEY_HASHES"))
        self.auth_disabled = os.getenv("FINPREDICT_AUTH_DISABLED", "").lower() == "true"
        self.rate_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
        self.max_failures_per_minute = int(os.getenv("AUTH_FAILURES_PER_MINUTE", "10"))
        self.limiter = RateLimiter()
        if not self.key_hashes and not self.auth_disabled:
            raise RuntimeError(
                "Aucune clé d'API configurée. Définissez FINPREDICT_API_KEY_HASHES "
                "(voir `python security.py gen-key NOM`) ; pour le développement local "
                "uniquement : FINPREDICT_AUTH_DISABLED=true."
            )

    def __call__(self, request: Request, x_api_key: str | None = Header(default=None)) -> str:
        ip = request.client.host if request.client else "unknown"
        if self.auth_disabled:
            caller = "dev-no-auth"
        else:
            allowed, retry = self.limiter.check(f"fail:{ip}", self.max_failures_per_minute, record=False)
            if not allowed:
                raise HTTPException(429, "Trop de tentatives invalides.", headers={"Retry-After": str(retry)})
            caller = find_key_name(x_api_key, self.key_hashes) if x_api_key else None
            if caller is None:
                self.limiter.check(f"fail:{ip}", self.max_failures_per_minute)  # enregistre l'échec
                raise HTTPException(401, "Clé d'API absente ou invalide.", headers={"WWW-Authenticate": "ApiKey"})

        allowed, retry = self.limiter.check(f"key:{caller}", self.rate_per_minute)
        if not allowed:
            raise HTTPException(429, "Limite de requêtes dépassée.", headers={"Retry-After": str(retry)})
        request.state.caller = caller
        return caller


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "gen-key":
        name = sys.argv[2] if len(sys.argv) > 2 else "client"
        key = secrets.token_urlsafe(32)
        print(f"Clé à transmettre au client (non récupérable ensuite) :\n  {key}\n")
        print("Ligne à ajouter / compléter dans .env :")
        print(f"  FINPREDICT_API_KEY_HASHES={name}:{hash_key(key)}")
    else:
        print("Usage : python security.py gen-key [nom]")
