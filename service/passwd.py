"""Password hash for the app's default account (APP_PASSWORD_HASH).

python -m service.passwd    asks for the password twice, prints scrypt:<salt hex>:<hash hex> (no $, safe in systemd env files)
"""
import getpass
import hashlib
import hmac
import secrets


def make(password: str, salt: bytes = None) -> str:
    salt = salt or secrets.token_bytes(16)
    return f"scrypt:{salt.hex()}:{hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1).hex()}"


def verify(password: str, stored: str) -> bool:
    try:
        kind, salt, _ = stored.split(":")
        return kind == "scrypt" and hmac.compare_digest(make(password, bytes.fromhex(salt)).encode(), stored.encode())
    except ValueError:
        return False


if __name__ == "__main__":
    pw = getpass.getpass("New password: ")
    if len(pw) < 8 or pw != getpass.getpass("Again: "):
        raise SystemExit("passwords differ or shorter than 8 characters")
    print(make(pw))
