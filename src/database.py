import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/aurapredict.db')


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS maquinas ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "nombre TEXT NOT NULL UNIQUE,"
        "tipo TEXT NOT NULL,"
        "descripcion TEXT,"
        "ubicacion TEXT,"
        "fecha_registro TEXT NOT NULL)"
    )

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS lecturas_rodamiento ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "timestamp TEXT NOT NULL,"
        "maquina TEXT NOT NULL,"
        "rms REAL,"
        "peak_to_peak REAL,"
        "kurtosis REAL,"
        "skewness REAL,"
        "resultado TEXT NOT NULL,"
        "nivel_riesgo TEXT NOT NULL)"
    )

    cursor.execute(
        "CREATE TABLE IF NOT EXISTS lecturas_prensa ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "timestamp TEXT NOT NULL,"
        "maquina TEXT NOT NULL,"
        "desviacion_columnas_ue REAL,"
        "vibracion_bomba_db REAL,"
        "particulas_aceite_iso INTEGER,"
        "resultado TEXT NOT NULL,"
        "nivel_riesgo TEXT NOT NULL)"
    )

    conn.commit()
    conn.close()


# --- GESTION DE MAQUINAS ---

def registrar_maquina(nombre, tipo, descripcion, ubicacion):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = ("INSERT INTO maquinas (nombre, tipo, descripcion, ubicacion, fecha_registro) "
           "VALUES (?, ?, ?, ?, ?)")
    valores = (nombre, tipo, descripcion, ubicacion,
               datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        cursor.execute(sql, valores)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # nombre duplicado
    finally:
        conn.close()


def obtener_maquinas():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT nombre, tipo, descripcion, ubicacion, fecha_registro "
        "FROM maquinas ORDER BY tipo, nombre"
    )
    filas = cursor.fetchall()
    conn.close()
    return filas


def obtener_maquina(nombre):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT nombre, tipo, descripcion, ubicacion, fecha_registro "
        "FROM maquinas WHERE nombre = ?",
        (nombre,)
    )
    fila = cursor.fetchone()
    conn.close()
    return fila


def eliminar_maquina(nombre):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM lecturas_rodamiento WHERE maquina = ?", (nombre,))
    cursor.execute("DELETE FROM lecturas_prensa WHERE maquina = ?", (nombre,))
    cursor.execute("DELETE FROM maquinas WHERE nombre = ?", (nombre,))
    conn.commit()
    conn.close()


# --- LECTURAS ---

def guardar_lectura_rodamiento(maquina, rms, peak_to_peak, kurtosis, skewness, resultado, nivel_riesgo, diagnostico=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Migración: añade la columna si no existe todavía
    try:
        cursor.execute(
            "ALTER TABLE lecturas_rodamiento ADD COLUMN diagnostico TEXT DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # La columna ya existe

    sql = ("INSERT INTO lecturas_rodamiento "
           "(timestamp, maquina, rms, peak_to_peak, kurtosis, skewness, resultado, nivel_riesgo, diagnostico) "
           "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)")
    valores = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               maquina, rms, peak_to_peak, kurtosis, skewness, resultado, nivel_riesgo, diagnostico)
    cursor.execute(sql, valores)
    conn.commit()
    conn.close()


def guardar_lectura_prensa(maquina, desviacion, vibracion, particulas, resultado, nivel_riesgo):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = ("INSERT INTO lecturas_prensa "
           "(timestamp, maquina, desviacion_columnas_ue, vibracion_bomba_db, "
           "particulas_aceite_iso, resultado, nivel_riesgo) "
           "VALUES (?, ?, ?, ?, ?, ?, ?)")
    valores = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               maquina, desviacion, vibracion, particulas, resultado, nivel_riesgo)
    cursor.execute(sql, valores)
    conn.commit()
    conn.close()


def obtener_historial_rodamiento(maquina, limite=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = ("SELECT timestamp, rms, kurtosis, peak_to_peak, resultado "
           "FROM lecturas_rodamiento WHERE maquina = ? "
           "ORDER BY id DESC LIMIT ?")
    cursor.execute(sql, (maquina, limite))
    filas = cursor.fetchall()
    conn.close()
    return list(reversed(filas))


def obtener_historial_prensa(maquina, limite=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    sql = ("SELECT timestamp, desviacion_columnas_ue, vibracion_bomba_db, "
           "particulas_aceite_iso, resultado "
           "FROM lecturas_prensa WHERE maquina = ? "
           "ORDER BY id DESC LIMIT ?")
    cursor.execute(sql, (maquina, limite))
    filas = cursor.fetchall()
    conn.close()
    return list(reversed(filas))


def contar_lecturas_rodamiento(maquina):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM lecturas_rodamiento WHERE maquina = ?",
        (maquina,)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count


init_db()
