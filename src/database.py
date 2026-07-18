import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    """Devuelve una conexión a PostgreSQL."""
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Crea las tablas si no existen."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS empresas (
            id              SERIAL PRIMARY KEY,
            nombre          TEXT NOT NULL UNIQUE,
            contacto        TEXT,
            email           TEXT,
            fecha_registro  TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id              SERIAL PRIMARY KEY,
            email           TEXT NOT NULL UNIQUE,
            password_hash   TEXT NOT NULL,
            nombre          TEXT,
            rol             TEXT NOT NULL DEFAULT 'cliente',
            empresa_id      INTEGER REFERENCES empresas(id),
            activo          BOOLEAN DEFAULT TRUE,
            fecha_registro  TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS maquinas (
            id              SERIAL PRIMARY KEY,
            nombre          TEXT NOT NULL UNIQUE,
            tipo            TEXT NOT NULL,
            descripcion     TEXT,
            ubicacion       TEXT,
            fecha_registro  TEXT NOT NULL,
            emails_alerta   TEXT DEFAULT '',
            empresa_id      INTEGER REFERENCES empresas(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lecturas_rodamiento (
            id           SERIAL PRIMARY KEY,
            timestamp    TEXT NOT NULL,
            maquina      TEXT NOT NULL,
            rms          REAL,
            peak_to_peak REAL,
            kurtosis     REAL,
            skewness     REAL,
            resultado    TEXT NOT NULL,
            nivel_riesgo TEXT NOT NULL,
            diagnostico  TEXT DEFAULT ''
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS lecturas_prensa (
            id                     SERIAL PRIMARY KEY,
            timestamp              TEXT NOT NULL,
            maquina                TEXT NOT NULL,
            desviacion_columnas_ue REAL,
            vibracion_bomba_db     REAL,
            particulas_aceite_iso  INTEGER,
            resultado              TEXT NOT NULL,
            nivel_riesgo           TEXT NOT NULL
        )
    """)

    # Migracion: Añadir empresa_id a maquinas
    try:
        cursor.execute(
            "ALTER TABLE maquinas ADD COLUMN empresa_id INTEGER REFERENCES empresas(id)"
        )
        conn.commit()
    except Exception:
        conn.rollback()

    conn.commit()
    cursor.close()
    conn.close()


# --- GESTION DE MAQUINAS ---

def obtener_maquinas():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT nombre, tipo, descripcion, ubicacion, fecha_registro "
        "FROM maquinas ORDER BY tipo, nombre"
    )
    filas = cursor.fetchall()
    cursor.close()
    conn.close()
    return filas


def obtener_maquina(nombre):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT nombre, tipo, descripcion, ubicacion, fecha_registro "
        "FROM maquinas WHERE nombre = %s",
        (nombre,)
    )
    fila = cursor.fetchone()
    cursor.close()
    conn.close()
    return fila


def obtener_emails_maquina(nombre):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT emails_alerta FROM maquinas WHERE nombre = %s", (nombre,))
    fila = cursor.fetchone()
    cursor.close()
    conn.close()
    if not fila or not fila[0]:
        return []
    return [e.strip() for e in fila[0].split(",") if e.strip()]


def eliminar_maquina(nombre):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM lecturas_rodamiento WHERE maquina = %s", (nombre,))
    cursor.execute("DELETE FROM lecturas_prensa WHERE maquina = %s", (nombre,))
    cursor.execute("DELETE FROM maquinas WHERE nombre = %s", (nombre,))
    conn.commit()
    cursor.close()
    conn.close()


# --- LECTURAS ---

def guardar_lectura_rodamiento(maquina, rms, peak_to_peak, kurtosis, skewness,
                               resultado, nivel_riesgo, diagnostico=""):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        INSERT INTO lecturas_rodamiento
        (timestamp, maquina, rms, peak_to_peak, kurtosis, skewness,
         resultado, nivel_riesgo, diagnostico)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    valores = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               maquina, rms, peak_to_peak, kurtosis, skewness,
               resultado, nivel_riesgo, diagnostico)
    cursor.execute(sql, valores)
    conn.commit()
    cursor.close()
    conn.close()


def guardar_lectura_prensa(maquina, desviacion, vibracion, particulas,
                           resultado, nivel_riesgo):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        INSERT INTO lecturas_prensa
        (timestamp, maquina, desviacion_columnas_ue, vibracion_bomba_db,
         particulas_aceite_iso, resultado, nivel_riesgo)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    valores = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               maquina, desviacion, vibracion, particulas,
               resultado, nivel_riesgo)
    cursor.execute(sql, valores)
    conn.commit()
    cursor.close()
    conn.close()


def obtener_historial_rodamiento(maquina, limite=50):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        SELECT timestamp, rms, kurtosis, peak_to_peak, resultado, diagnostico
        FROM lecturas_rodamiento
        WHERE maquina = %s
        ORDER BY id DESC
        LIMIT %s
    """
    cursor.execute(sql, (maquina, limite))
    filas = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(reversed(filas))


def obtener_historial_prensa(maquina, limite=50):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        SELECT timestamp, desviacion_columnas_ue, vibracion_bomba_db,
               particulas_aceite_iso, resultado
        FROM lecturas_prensa
        WHERE maquina = %s
        ORDER BY id DESC
        LIMIT %s
    """
    cursor.execute(sql, (maquina, limite))
    filas = cursor.fetchall()
    cursor.close()
    conn.close()
    return list(reversed(filas))


def contar_lecturas_rodamiento(maquina):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM lecturas_rodamiento WHERE maquina = %s",
        (maquina,)
    )
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count

# --- GESTIÓN DE EMPRESAS ---


def crear_empresa(nombre, contacto="", email=""):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        INSERT INTO empresas (nombre, contacto, email, fecha_registro)
        VALUES (%s, %s, %s, %s) RETURNING id
    """
    valores = (nombre, contacto, email,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        cursor.execute(sql, valores)
        empresa_id = cursor.fetchone()[0]
        conn.commit()
        return empresa_id
    except psycopg2.IntegrityError:
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()


def obtener_empresas():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, nombre, contacto, email, fecha_registro FROM empresas ORDER BY nombre"
    )
    filas = cursor.fetchall()
    cursor.close()
    conn.close()
    return filas


def obtener_empresa_por_id(empresa_id):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, nombre, contacto, email FROM empresas WHERE id = %s",
        (empresa_id,)
    )
    fila = cursor.fetchone()
    cursor.close()
    conn.close()
    return fila


# --- GESTIÓN DE USUARIOS ---

def crear_usuario(email, password_hash, nombre, rol, empresa_id=None):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        INSERT INTO usuarios (email, password_hash, nombre, rol, empresa_id, fecha_registro)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    valores = (email, password_hash, nombre, rol, empresa_id,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        cursor.execute(sql, valores)
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def obtener_usuario_por_email(email):
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, email, password_hash, nombre, rol, empresa_id, activo "
        "FROM usuarios WHERE email = %s",
        (email,)
    )
    fila = cursor.fetchone()
    cursor.close()
    conn.close()
    return fila


def obtener_usuarios():
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT u.id, u.email, u.nombre, u.rol, u.activo,
               e.nombre as empresa
        FROM usuarios u
        LEFT JOIN empresas e ON u.empresa_id = e.id
        ORDER BY u.rol, u.nombre
    """)
    filas = cursor.fetchall()
    cursor.close()
    conn.close()
    return filas


# --- MÁQUINAS CON FILTRO POR EMPRESA ---

def obtener_maquinas_por_empresa(empresa_id=None):
    """Si empresa_id es None devuelve todas (para admin)."""
    conn = get_conn()
    cursor = conn.cursor()
    if empresa_id is None:
        cursor.execute(
            "SELECT nombre, tipo, descripcion, ubicacion, fecha_registro "
            "FROM maquinas ORDER BY tipo, nombre"
        )
    else:
        cursor.execute(
            "SELECT nombre, tipo, descripcion, ubicacion, fecha_registro "
            "FROM maquinas WHERE empresa_id = %s ORDER BY tipo, nombre",
            (empresa_id,)
        )
    filas = cursor.fetchall()
    cursor.close()
    conn.close()
    return filas


def asignar_empresa_maquina(nombre_maquina, empresa_id):
    """Asigna una máquina a una empresa."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE maquinas SET empresa_id = %s WHERE nombre = %s",
        (empresa_id, nombre_maquina)
    )
    conn.commit()
    cursor.close()
    conn.close()


def registrar_maquina(nombre, tipo, descripcion, ubicacion,
                      emails_alerta="", empresa_id=None):
    conn = get_conn()
    cursor = conn.cursor()
    sql = """
        INSERT INTO maquinas
        (nombre, tipo, descripcion, ubicacion, fecha_registro, emails_alerta, empresa_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    valores = (nombre, tipo, descripcion, ubicacion,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               emails_alerta, empresa_id)
    try:
        cursor.execute(sql, valores)
        conn.commit()
        return True
    except psycopg2.IntegrityError:
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()


def obtener_ultima_lectura(nombre_maquina, tipo):
    """Obtiene el resultado de la última lectura de una máquina."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        tabla = "lecturas_rodamiento" if tipo == "rodamiento" else "lecturas_prensa"
        cur.execute(f"""
            SELECT resultado, nivel_riesgo, timestamp
            FROM {tabla}
            WHERE maquina = %s
            ORDER BY timestamp DESC LIMIT 1
        """, (nombre_maquina,))
        return cur.fetchone()
    except Exception as e:
        print(f"Error obteniendo última lectura: {e}")
        return None
    finally:
        cur.close()
        conn.close()


init_db()
