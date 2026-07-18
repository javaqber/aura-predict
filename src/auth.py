import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

SECRET_KEY = os.getenv("SECRET_KEY", "cambia_esto_por_una_clave_segura")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HORAS = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hashear_password(password: str) -> str:
    """Convierte una contraseña en texto plano a hash bcrypt."""
    return pwd_context.hash(password)


def verificar_password(password: str, hash: str) -> bool:
    """Comprueba si una contraseña coincide con su hash."""
    return pwd_context.verify(password, hash)


def crear_token(datos: dict) -> str:
    """Genera un JWT con los datos del usuario y fecha de expiración."""
    payload = datos.copy()
    expira = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HORAS)
    payload.update({"exp": expira})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verificar_token(token: str) -> dict | None:
    """Verifica un JWT y devuelve los datos si es válido, None si no."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
