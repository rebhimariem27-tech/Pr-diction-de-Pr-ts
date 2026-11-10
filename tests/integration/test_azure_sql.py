import os
import sys

from dotenv import load_dotenv
from backend.app.sql_compat import connect


# 1. Charger les paramètres du fichier .env
load_dotenv()

server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DATABASE")
username = os.getenv("SQL_USERNAME")
password = os.getenv("SQL_PASSWORD")

missing = [
    name
    for name, value in {
        "SQL_SERVER": server,
        "SQL_DATABASE": database,
        "SQL_USERNAME": username,
        "SQL_PASSWORD": password,
    }.items()
    if not value
]

if missing:
    sys.exit(
        "Configuration incomplète dans .env : "
        + ", ".join(missing)
    )


# 2. Protéger les caractères spéciaux des identifiants
def quote_value(value: str) -> str:
    if any(char in value for char in ";={}"):
        return "{" + value.replace("}", "}}") + "}"
    return value


# 3. Préparer une connexion chiffrée vers Azure SQL
connection_string = (
    f"Server={quote_value(server)};"
    f"Database={quote_value(database)};"
    f"UID={quote_value(username)};"
    f"PWD={quote_value(password)};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
)


# 4. Tester la connexion et interroger la base distante
def main():
    print("Connexion à Azure SQL en cours...")

    try:
        with connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        DB_NAME(),
                        SUSER_SNAME(),
                        SYSDATETIMEOFFSET();
                    """
                )

                row = cursor.fetchone()

                print("\nConnexion réussie !")
                print(f"Base active : {row[0]}")
                print(f"Compte connecté : {row[1]}")
                print(f"Heure du serveur : {row[2]}")

    except Exception as exc:
        print("\nÉchec de la connexion.")
        print(f"Type d'erreur : {type(exc).__name__}")
        print(f"Détail : {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()