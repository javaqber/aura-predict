from database import (
    guardar_lectura_rodamiento, guardar_lectura_prensa,
    obtener_historial_rodamiento, obtener_historial_prensa,
    contar_lecturas_rodamiento,
    registrar_maquina, obtener_maquinas, obtener_maquina, eliminar_maquina
)
import joblib
import pandas as pd
import streamlit as st
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# --- CONFIG ---
st.set_page_config(page_title="AuraPredict", page_icon="⚙️", layout="wide")
LECTURAS_MINIMAS = 20

# --- SESSION STATE ---
if "maquina_activa" not in st.session_state:
    st.session_state.maquina_activa = None
if "confirmar_eliminar" not in st.session_state:
    st.session_state.confirmar_eliminar = None

# --- MODELO ---


@st.cache_resource
def cargar_modelo():
    ruta = os.path.join(os.path.dirname(__file__),
                        '../models/rodamientos_mecanizado.joblib')
    if os.path.exists(ruta):
        return joblib.load(ruta)
    ruta_alt = "models/rodamientos_mecanizado.joblib"
    if os.path.exists(ruta_alt):
        return joblib.load(ruta_alt)
    return None


modelo_ia = cargar_modelo()

# --- CSS ---
st.markdown("""
    <style>
    button { cursor: pointer !important; }
    .card-maquina {
        background: #F8F9FA;
        border: 1px solid #DEE2E6;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .card-activa {
        background: #E3F2FD;
        border: 2px solid #0B3D91;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## ⚙️ AuraPredict")
    st.markdown("---")
    st.markdown("### 🏭 Mis Máquinas")

    # --- FORMULARIO NUEVA MÁQUINA ---
    with st.expander("➕ Registrar nueva máquina"):
        with st.form("form_nueva"):
            n_nombre = st.text_input(
                "Nombre / ID *", placeholder="Ej: Torno_CNC_1")
            n_tipo = st.selectbox("Tipo *", ["rodamiento", "prensa"],
                                  help="Rodamiento: tornos y fresadoras | Prensa: prensas hidráulicas")
            n_desc = st.text_input(
                "Descripción", placeholder="Ej: Torno 2 ejes, Zona A")
            n_ubic = st.text_input("Ubicación en planta",
                                   placeholder="Ej: Nave 2, Línea 3")
            guardar = st.form_submit_button(
                "Registrar", use_container_width=True)

            if guardar:
                if not n_nombre.strip():
                    st.error("El nombre es obligatorio.")
                else:
                    ok = registrar_maquina(
                        n_nombre.strip(), n_tipo, n_desc, n_ubic)
                    if ok:
                        st.success(f"'{n_nombre}' registrada correctamente.")
                        st.rerun()
                    else:
                        st.error("Ya existe una máquina con ese nombre.")

    st.markdown("")

    # --- LISTA DE MÁQUINAS ---
    maquinas = obtener_maquinas()
    EMOJI = {"rodamiento": "⚙️", "prensa": "🏗️"}

    if not maquinas:
        st.info("Registra tu primera máquina para empezar.")
    else:
        for m in maquinas:
            nombre, tipo, desc, ubic, fecha = m
            es_activa = st.session_state.maquina_activa == nombre
            emoji = EMOJI.get(tipo, "🔧")
            clase = "card-activa" if es_activa else "card-maquina"

            st.markdown(f"""
                <div class="{clase}">
                    <strong>{emoji} {nombre}</strong><br>
                    <small style="color:#666;">{desc if desc else tipo.capitalize()}</small>
                    {('<br><small style="color:#999;">📍 ' + ubic + '</small>') if ubic else ''}
                </div>
            """, unsafe_allow_html=True)

            col_sel, col_del = st.columns([3, 1])
            with col_sel:
                if es_activa:
                    st.button("✓ Activa", key=f"act_{nombre}",
                              disabled=True, use_container_width=True)
                else:
                    if st.button("Seleccionar", key=f"sel_{nombre}",
                                 use_container_width=True):
                        st.session_state.maquina_activa = nombre
                        st.session_state.confirmar_eliminar = None
                        st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_{nombre}", help=f"Eliminar {nombre}"):
                    st.session_state.confirmar_eliminar = nombre

    # --- CONFIRMACIÓN DE ELIMINACIÓN ---
    if st.session_state.confirmar_eliminar:
        nombre_elim = st.session_state.confirmar_eliminar
        st.markdown("---")
        st.warning(
            f"¿Eliminar **{nombre_elim}** y todo su historial? Esta acción no se puede deshacer.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Sí, eliminar", type="primary", use_container_width=True):
                eliminar_maquina(nombre_elim)
                if st.session_state.maquina_activa == nombre_elim:
                    st.session_state.maquina_activa = None
                st.session_state.confirmar_eliminar = None
                st.rerun()
        with c2:
            if st.button("Cancelar", use_container_width=True):
                st.session_state.confirmar_eliminar = None
                st.rerun()

# ================================================================
# MAIN — TÍTULO
# ================================================================
st.markdown("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <h1 style="font-size:3.5rem; font-weight:800; margin-bottom:0; color:#0A192F;">
            <span style="color:#0B3D91;">Aura</span>Predict
        </h1>
        <p style="font-size:1.1rem; color:#666; margin-top:-5px;">
            Monitorización Predictiva para Activos Industriales Críticos
        </p>
    </div>
""", unsafe_allow_html=True)
st.markdown("---")

# ================================================================
# MAIN — SIN MÁQUINA SELECCIONADA
# ================================================================
if st.session_state.maquina_activa is None:
    st.markdown("""
        <div style="text-align:center; padding:60px 0;">
            <h3 style="color:#666;">👈 Selecciona o registra una máquina desde el panel lateral</h3>
            <p style="color:#999;">Cada máquina mantiene su propio historial independiente.</p>
        </div>
    """, unsafe_allow_html=True)

# ================================================================
# MAIN — CON MÁQUINA SELECCIONADA
# ================================================================
else:
    datos_maquina = obtener_maquina(st.session_state.maquina_activa)
    if datos_maquina is None:
        st.error("La máquina activa ya no existe.")
        st.session_state.maquina_activa = None
        st.rerun()

    nombre, tipo, desc, ubic, fecha = datos_maquina
    TIPO_LABEL = {"rodamiento": "Torno CNC / Fresadora",
                  "prensa": "Prensa de Extrusión"}

    # Ficha de la máquina
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Máquina", f"{EMOJI.get(tipo, '🔧')} {nombre}")
    c2.metric("Tipo", TIPO_LABEL.get(tipo, tipo))
    c3.metric("Ubicación", ubic or "—")
    c4.metric("Registrada", fecha[:10])
    if desc:
        st.caption(f"📋 {desc}")

    st.markdown("---")

    # Indicador de fase (solo rodamiento)
    if tipo == "rodamiento":
        total = contar_lecturas_rodamiento(nombre)
        if total < LECTURAS_MINIMAS:
            st.markdown(f"""
                <div style="background:#FFF8E1;border-left:5px solid #F9A825;
                            padding:12px 20px;border-radius:4px;margin-bottom:16px;">
                    🟡 <strong>FASE DE APRENDIZAJE</strong> —
                    Lecturas acumuladas: <strong>{total} / {LECTURAS_MINIMAS}</strong>.
                    El diagnóstico gana precisión con cada análisis.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background:#E8F5E9;border-left:5px solid #2E7D32;
                            padding:12px 20px;border-radius:4px;margin-bottom:16px;">
                    🟢 <strong>MONITORIZACIÓN ACTIVA</strong> —
                    Modelo calibrado con <strong>{total}</strong> lecturas históricas.
                </div>
            """, unsafe_allow_html=True)

    # Formulario de entrada de datos
    if tipo == "rodamiento":
        st.subheader("📡 Sensor de Análisis de Vibración")
        col1, col2 = st.columns(2)
        with col1:
            rms_val = st.number_input(
                "RMS (Fuerza)", value=0.07, format="%.4f")
            p2p_val = st.number_input(
                "Peak-to-Peak", value=0.84, format="%.4f")
        with col2:
            kurt_val = st.number_input(
                "Kurtosis (Impactos)", value=0.62, format="%.4f")
            skew_val = st.number_input("Skewness", value=0.08, format="%.4f")
    else:
        st.subheader("🏗️ Monitorización Estructural e Hidráulica")
        st.caption("Sensores IoT independientes del HMI del operario")
        col1, col2, col3 = st.columns(3)
        with col1:
            columnas = st.number_input("Desviación Columnas (µε)", value=2.5, format="%.1f",
                                       help="Diferencia de tensión entre columnas")
        with col2:
            bomba = st.number_input("Firma Acústica Bomba (dB)", value=65.0, format="%.1f",
                                    help="Ultrasonidos en bomba principal")
        with col3:
            aceite = st.number_input("Partículas Aceite (ISO)", value=14, format="%d",
                                     help="Contaminación sólida en tanque")

    st.markdown("---")

    # Botón de análisis
    if st.button("🔍 Analizar Estado del Activo", use_container_width=True):
        if tipo == "rodamiento":
            if modelo_ia is None:
                st.error(
                    "🚨 ERROR: Modelo no encontrado. Revisa la carpeta models/.")
            else:
                prediccion = modelo_ia.predict(pd.DataFrame([{
                    "RMS": rms_val, "Peak_to_Peak": p2p_val,
                    "Kurtosis": kurt_val, "Skewness": skew_val
                }]))[0]

                from diagnostico import diagnosticar_rodamiento
                diag = diagnosticar_rodamiento(
                    rms_val, p2p_val, kurt_val, skew_val)

                if prediccion == 1:
                    estado = "OK - Sano"
                    riesgo = "Bajo"
                    st.success("✅ ESTADO: OK — Rodamiento en buen estado")
                    st.info("🟢 RIESGO: Bajo")
                else:
                    estado = "NOK - Anomalía Detectada"
                    riesgo = "CRÍTICO - Parar Máquina"
                    st.error("🚨 ESTADO: NOK — Anomalía Detectada")
                    st.warning("⚠️ RIESGO: CRÍTICO — Parar Máquina")

                # Tarjeta de diagnóstico
                COLORES = {
                    "verde":   ("#E8F5E9", "#2E7D32"),
                    "amarillo": ("#FFFDE7", "#F9A825"),
                    "naranja": ("#FFF3E0", "#E65100"),
                    "rojo":    ("#FFEBEE", "#C62828")
                }
                fondo, borde = COLORES.get(
                    diag["nivel_urgencia"], ("#F5F5F5", "#666"))

                st.markdown(f"""
                    <div style="background:{fondo};border-left:5px solid {borde};
                                padding:16px 20px;border-radius:6px;margin-top:16px;">
                        <h4 style="color:{borde};margin-top:0;">
                            🔎 Diagnóstico: {diag['tipo_fallo']}
                        </h4>
                        <p style="margin:4px 0;"><strong>Componente afectado:</strong> {diag['componente_afectado']}</p>
                        <p style="margin:4px 0;"><strong>Descripción:</strong> {diag['descripcion']}</p>
                        <p style="margin:4px 0;"><strong>Consecuencias:</strong> {diag['consecuencias']}</p>
                        <p style="margin:4px 0;"><strong>Acción recomendada:</strong> {diag['accion_recomendada']}</p>
                        <p style="margin:4px 0;"><strong>Pieza / referencia:</strong> {diag['pieza_referencia']}</p>
                        <p style="margin:4px 0;"><strong>Ventana de actuación:</strong>
                            <span style="color:{borde};font-weight:bold;">{diag['ventana_actuacion']}</span>
                        </p>
                        <p style="margin:4px 0;color:#888;font-size:0.85em;">
                        Confianza del diagnóstico: {diag['confianza']}
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                guardar_lectura_rodamiento(
                    maquina=nombre, rms=rms_val, peak_to_peak=p2p_val,
                    kurtosis=kurt_val, skewness=skew_val,
                    resultado=estado, nivel_riesgo=riesgo,
                    diagnostico=diag["tipo_fallo"]
                )

                st.caption("✔️ Lectura registrada en el historial.")

        else:
            if columnas > 15.0:
                estado = "NOK - Desalineación Estructural"
                riesgo = "CRÍTICO - Peligro de rotura de columna"
                st.error("🚨 ESTADO: NOK — Desalineación Estructural")
                st.warning("⚠️ RIESGO: CRÍTICO — Peligro de rotura de columna")
            elif bomba > 85.0 or aceite > 18:
                estado = "NOK - Desgaste Hidráulico"
                riesgo = "ALTO - Filtrar aceite / Revisar bomba"
                st.error("🚨 ESTADO: NOK — Desgaste Hidráulico")
                st.warning("⚠️ RIESGO: ALTO — Filtrar aceite / Revisar bomba")
            else:
                estado = "OK - Sanos"
                riesgo = "Bajo"
                st.success(
                    "✅ ESTADO: OK — Estructura y sistema hidráulico sanos")
                st.info("🟢 RIESGO: Bajo")

            guardar_lectura_prensa(
                maquina=nombre, desviacion=columnas, vibracion=bomba,
                particulas=aceite, resultado=estado, nivel_riesgo=riesgo
            )
            st.caption("✔️ Lectura registrada en el historial.")

    # Historial gráfico
    st.markdown("---")
    st.subheader("📈 Historial de Lecturas")

    if tipo == "rodamiento":
        historial = obtener_historial_rodamiento(nombre, limite=50)
        if not historial:
            st.info(
                "Aún no hay lecturas para esta máquina. Realiza tu primer análisis.")
        else:
            df = pd.DataFrame(historial,
                              columns=["Timestamp", "RMS", "Kurtosis", "Peak_to_Peak", "Resultado"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            cg1, cg2 = st.columns(2)
            with cg1:
                st.markdown("**RMS — Evolución temporal**")
                st.line_chart(df.set_index("Timestamp")["RMS"])
            with cg2:
                st.markdown("**Kurtosis — Evolución temporal**")
                st.line_chart(df.set_index("Timestamp")["Kurtosis"])
            with st.expander("Ver tabla completa"):
                st.dataframe(df, use_container_width=True)
    else:
        historial = obtener_historial_prensa(nombre, limite=50)
        if not historial:
            st.info(
                "Aún no hay lecturas para esta máquina. Realiza tu primer análisis.")
        else:
            df = pd.DataFrame(historial, columns=[
                "Timestamp", "Desviacion_uE", "Vibracion_dB", "Particulas_ISO", "Resultado"])
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            cg1, cg2, cg3 = st.columns(3)
            with cg1:
                st.markdown("**Desviación Columnas (µε)**")
                st.line_chart(df.set_index("Timestamp")["Desviacion_uE"])
            with cg2:
                st.markdown("**Vibración Bomba (dB)**")
                st.line_chart(df.set_index("Timestamp")["Vibracion_dB"])
            with cg3:
                st.markdown("**Partículas Aceite (ISO)**")
                st.line_chart(df.set_index("Timestamp")["Particulas_ISO"])
            with st.expander("Ver tabla completa"):
                st.dataframe(df, use_container_width=True)
