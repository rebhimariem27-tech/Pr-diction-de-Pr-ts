"""Génère une clé d'API et l'installe des deux côtés, sans copier-coller.

    python -m backend.scripts.setup_api_key

- Backend  : écrit/remplace FINPREDICT_API_KEY_HASHES=frontend:<hash> dans le .env du backend.
- Frontend : écrit/remplace FINPREDICT_API_KEY=<clé> dans finpredict-frontend/.env.local
             (et ajoute FINPREDICT_API_URL / NEXT_PUBLIC_API_BASE_URL s'ils manquent).
Les autres lignes de ces fichiers sont conservées. La clé n'est JAMAIS affichée.

Après exécution : redémarrer uvicorn ET `npm run dev` (les deux ne lisent leur .env qu'au démarrage).
Options : --backend-env CHEMIN  --frontend-env CHEMIN  --name NOM
"""
from __future__ import annotations

import argparse
import os
import secrets
from pathlib import Path

from backend.app.security import hash_key

ROOT = Path(__file__).resolve().parents[2]


def upsert(path: Path, updates: dict[str, str], only_if_missing: set[str] = frozenset()) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    done, changed = set(), []
    for i, line in enumerate(lines):
        name = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and name in updates:
            if name in only_if_missing:
                done.add(name)
                continue
            if name in done:          # doublon : on neutralise les anciennes occurrences
                lines[i] = f"# (remplacé) {line}"
                continue
            lines[i] = f"{name}={updates[name]}"
            done.add(name)
            changed.append(f"{name} (remplacée)")
    for name, value in updates.items():
        if name not in done:
            lines.append(f"{name}={value}")
            changed.append(f"{name} (ajoutée)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))   # UTF-8 sans BOM
    return changed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--backend-env", type=Path,
                   default=Path(os.getenv("FINPREDICT_ENV_FILE") or ROOT / ".env"))
    p.add_argument("--frontend-env", type=Path, default=ROOT / "finpredict-frontend" / ".env.local")
    p.add_argument("--name", default="frontend")
    a = p.parse_args()

    if not a.frontend_env.parent.is_dir():
        raise SystemExit(f"Dossier frontend introuvable : {a.frontend_env.parent}")

    key = secrets.token_urlsafe(32)
    back = upsert(a.backend_env, {"FINPREDICT_API_KEY_HASHES": f"{a.name}:{hash_key(key)}"})
    front = upsert(a.frontend_env, {
        "FINPREDICT_API_KEY": key,
        "FINPREDICT_API_URL": "http://127.0.0.1:8000",
        "NEXT_PUBLIC_API_BASE_URL": "http://127.0.0.1:8000",
    }, only_if_missing={"FINPREDICT_API_URL", "NEXT_PUBLIC_API_BASE_URL"})

    print(f"Backend  : {a.backend_env}\n   " + ", ".join(back))
    print(f"Frontend : {a.frontend_env}\n   " + ", ".join(front))
    print(f"\nClé générée ({len(key)} caractères), non affichée.")
    print("Maintenant : redémarrez uvicorn ET `npm run dev`.")


if __name__ == "__main__":
    main()
