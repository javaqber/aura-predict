from diagnostico import diagnosticar_rodamiento
from database import (
    guardar_lectura_rodamiento, guardar_lectura_prensa,
    obtener_historial_rodamiento, obtener_historial_prensa,
    contar_lecturas_rodamiento,
    registrar_maquina, obtener_maquinas, obtener_maquina, eliminar_maquina,
    obtener_maquinas_por_empresa, obtener_empresas
)
import joblib
import pandas as pd
import streamlit as st
import requests
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))


# --- CONFIG ---
st.set_page_config(page_title="AuraPredict", page_icon="⚙️", layout="wide")

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
LECTURAS_MINIMAS = 20
EMOJI = {"rodamiento": "⚙️", "prensa": "🏗️"}

# --- SESSION STATE ---
for key, val in {
    "token": None, "usuario": None, "rol": None, "empresa_id": None,
    "maquina_activa": None, "confirmar_eliminar": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

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
        background: #F8F9FA; border: 1px solid #DEE2E6;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
    }
    .card-activa {
        background: #E3F2FD; border: 2px solid #0B3D91;
        border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOGIN ---


def hacer_login(email, password):
    try:
        r = requests.post(
            f"{API_BASE_URL}/login",
            json={"email": email, "password": password},
            timeout=10
        )
        if r.status_code == 200:
            datos = r.json()
            st.session_state.token = datos["access_token"]
            st.session_state.usuario = datos["nombre"]
            st.session_state.rol = datos["rol"]
            st.session_state.empresa_id = datos["empresa_id"]
            return True, None
        return False, r.json().get("detail", "Error desconocido")
    except Exception as e:
        return False, f"No se pudo conectar con la API: {e}"


# ================================================================
# PANTALLA DE LOGIN
# ================================================================
if st.session_state.token is None:
    st.markdown("""
        <div style="max-width:400px;margin:80px auto 0 auto;text-align:center;">
            <h1 style="font-size:3rem;font-weight:800;color:#0A192F;">
                <span style="color:#0B3D91;">Aura</span>Predict
            </h1>
            <p style="color:#666;margin-bottom:32px;">
                Monitorización Predictiva Industrial
            </p>
        </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.form("form_login"):
            st.markdown("#### Iniciar sesión")
            email = st.text_input("Email", placeholder="tu@email.com")
            password = st.text_input("Contraseña", type="password")
            entrar = st.form_submit_button("Entrar", use_container_width=True)
            if entrar:
                if not email or not password:
                    st.error("Introduce email y contraseña.")
                else:
                    ok, error = hacer_login(email, password)
                    if ok:
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
    st.stop()

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    # Usuario y cerrar sesión
    col_u, col_s = st.columns([3, 1])
    with col_u:
        st.markdown(f"👤 **{st.session_state.usuario}**")
        st.caption(f"Rol: {st.session_state.rol}")
    with col_s:
        if st.button("🚪", help="Cerrar sesión"):
            for key in ["token", "usuario", "rol", "empresa_id",
                        "maquina_activa", "confirmar_eliminar"]:
                st.session_state[key] = None
            st.rerun()

    st.markdown("---")
    st.markdown("### 🏭 Mis Máquinas")

    # Formulario nueva máquina
    with st.expander("➕ Registrar nueva máquina"):
        with st.form("form_nueva"):
            n_nombre = st.text_input(
                "Nombre / ID *", placeholder="Ej: Torno_CNC_1")
            n_tipo = st.selectbox("Tipo *", ["rodamiento", "prensa"])
            n_desc = st.text_input(
                "Descripción", placeholder="Ej: Torno 2 ejes, Zona A")
            n_ubic = st.text_input("Ubicación en planta",
                                   placeholder="Ej: Nave 2, Línea 3")
            n_emails = st.text_input(
                "Emails de alerta",
                placeholder="cliente@empresa.com, responsable@empresa.com",
                help="Separados por coma."
            )
            guardar = st.form_submit_button(
                "Registrar", use_container_width=True)
            if guardar:
                if not n_nombre.strip():
                    st.error("El nombre es obligatorio.")
                else:
                    empresa_id = st.session_state.empresa_id
                    ok = registrar_maquina(
                        n_nombre.strip(), n_tipo, n_desc, n_ubic,
                        n_emails, empresa_id
                    )
                    if ok:
                        st.success(f"'{n_nombre}' registrada correctamente.")
                        st.rerun()
                    else:
                        st.error("Ya existe una máquina con ese nombre.")

    st.markdown("")

    # Lista de máquinas filtrada por empresa
    es_admin = st.session_state.rol == "admin"
    empresa_id = st.session_state.empresa_id
    maquinas = obtener_maquinas_por_empresa(None if es_admin else empresa_id)

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

    # Confirmación de eliminación
    if st.session_state.confirmar_eliminar:
        nombre_elim = st.session_state.confirmar_eliminar
        st.markdown("---")
        st.warning(f"¿Eliminar **{nombre_elim}** y todo su historial?")
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

    # --- PANEL ADMIN ---
    if st.session_state.rol == "admin":
        st.markdown("---")
        st.markdown("### ⚙️ Administración")

        tab_empresas, tab_usuarios = st.tabs(["Empresas", "Usuarios"])

        with tab_empresas:
            st.markdown("**Empresas registradas:**")
            empresas = obtener_empresas()
            if not empresas:
                st.info("No hay empresas registradas.")
            else:
                for emp in empresas:
                    id_emp, nombre_emp, contacto, email_emp, fecha_emp = emp
                    st.markdown(f"""
                        <div class="card-maquina">
                            <strong>🏢 {nombre_emp}</strong><br>
                            <small style="color:#666;">
                                {contacto or '—'} · {email_emp or '—'}
                            </small>
                        </div>
                    """, unsafe_allow_html=True)

            st.markdown("**Registrar nueva empresa:**")
            with st.form("form_empresa"):
                e_nombre = st.text_input("Nombre empresa *")
                e_contacto = st.text_input("Persona de contacto")
                e_email = st.text_input("Email de contacto")
                if st.form_submit_button("Crear empresa", use_container_width=True):
                    if not e_nombre.strip():
                        st.error("El nombre es obligatorio.")
                    else:
                        from database import crear_empresa
                        id_nueva = crear_empresa(
                            e_nombre.strip(), e_contacto, e_email)
                        if id_nueva:
                            st.success(
                                f"Empresa '{e_nombre}' creada con ID {id_nueva}.")
                            st.rerun()
                        else:
                            st.error("Ya existe una empresa con ese nombre.")

        with tab_usuarios:
            st.markdown("**Crear usuario cliente:**")
            empresas = obtener_empresas()
            if not empresas:
                st.warning("Crea primero una empresa.")
            else:
                with st.form("form_usuario"):
                    u_email = st.text_input("Email del usuario *")
                    u_nombre = st.text_input("Nombre completo *")
                    u_password = st.text_input(
                        "Contraseña temporal *", type="password")
                    u_empresa = st.selectbox(
                        "Empresa *",
                        options=[(e[0], e[1]) for e in empresas],
                        format_func=lambda x: x[1]
                    )
                    if st.form_submit_button("Crear usuario", use_container_width=True):
                        if not u_email.strip() or not u_nombre.strip() or not u_password:
                            st.error("Todos los campos son obligatorios.")
                        else:
                            from database import crear_usuario
                            from auth import hashear_password
                            ok = crear_usuario(
                                email=u_email.strip(),
                                password_hash=hashear_password(u_password),
                                nombre=u_nombre.strip(),
                                rol="cliente",
                                empresa_id=u_empresa[0]
                            )
                            if ok:
                                st.success(
                                    f"Usuario '{u_email}' creado correctamente. "
                                    f"Empresa: {u_empresa[1]}."
                                )
                            else:
                                st.error("Ya existe un usuario con ese email.")

            st.markdown("**Usuarios existentes:**")
            from database import obtener_usuarios
            usuarios = obtener_usuarios()
            if usuarios:
                for u in usuarios:
                    id_u, email_u, nombre_u, rol_u, activo_u, empresa_u = u
                    color = "#E8F5E9" if activo_u else "#FFEBEE"
                    st.markdown(f"""
                        <div style="background:{color};border-radius:6px;
                                    padding:8px 12px;margin-bottom:4px;">
                            <strong>{nombre_u}</strong>
                            <span style="color:#666;font-size:0.85em;">
                                · {email_u} · {rol_u} · {empresa_u or 'Sin empresa'}
                            </span>
                        </div>
                    """, unsafe_allow_html=True)

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

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Máquina", f"{EMOJI.get(tipo, '🔧')} {nombre}")
    c2.metric("Tipo", TIPO_LABEL.get(tipo, tipo))
    c3.metric("Ubicación", ubic or "—")
    c4.metric("Registrada", fecha[:10])
    if desc:
        st.caption(f"📋 {desc}")

    st.markdown("---")

    # Indicador de fase
    if tipo == "rodamiento":
        total = contar_lecturas_rodamiento(nombre)
        if total < LECTURAS_MINIMAS:
            st.markdown(f"""
                <div style="background:#FFF8E1;border-left:5px solid #F9A825;
                            padding:12px 20px;border-radius:4px;margin-bottom:16px;">
                    🟡 <strong>FASE DE APRENDIZAJE</strong> —
                    Lecturas acumuladas: <strong>{total} / {LECTURAS_MINIMAS}</strong>.
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

    # Formulario de entrada
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
            columnas = st.number_input(
                "Desviación Columnas (µε)", value=2.5, format="%.1f")
        with col2:
            bomba = st.number_input(
                "Firma Acústica Bomba (dB)", value=65.0, format="%.1f")
        with col3:
            aceite = st.number_input(
                "Partículas Aceite (ISO)", value=14, format="%d")

    st.markdown("---")

    # Botón de análisis
    if st.button("🔍 Analizar Estado del Activo", use_container_width=True):
        if tipo == "rodamiento":
            if modelo_ia is None:
                st.error("🚨 ERROR: Modelo no encontrado.")
            else:
                prediccion = modelo_ia.predict(pd.DataFrame([{
                    "RMS": rms_val, "Peak_to_Peak": p2p_val,
                    "Kurtosis": kurt_val, "Skewness": skew_val
                }]))[0]

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
            st.info("Aún no hay lecturas para esta máquina.")
        else:
            df = pd.DataFrame(historial,
                              columns=["Timestamp", "RMS", "Kurtosis",
                                       "Peak_to_Peak", "Resultado", "Diagnostico"])
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
            # --- EXPORTACIÓN EXCEL RODAMIENTO ---
            st.markdown("---")
            st.subheader("📥 Exportar Historial")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                fecha_inicio = st.date_input(
                    "Desde",
                    value=pd.Timestamp.now() - pd.Timedelta(days=30),
                    key="fecha_ini_rod"
                )
            with col_f2:
                fecha_fin = st.date_input(
                    "Hasta",
                    value=pd.Timestamp.now(),
                    key="fecha_fin_rod"
                )

            if st.button("Generar Excel", key="excel_rod", use_container_width=True):
                historial_completo = obtener_historial_rodamiento(
                    nombre, limite=10000)
                if not historial_completo:
                    st.warning("No hay datos para exportar.")
                else:
                    df_export = pd.DataFrame(
                        historial_completo,
                        columns=["Timestamp", "RMS", "Kurtosis",
                                 "Peak_to_Peak", "Resultado", "Diagnostico"]
                    )
                    df_export["Timestamp"] = pd.to_datetime(
                        df_export["Timestamp"])
                    mask = (
                        (df_export["Timestamp"].dt.date >= fecha_inicio) &
                        (df_export["Timestamp"].dt.date <= fecha_fin)
                    )
                    df_filtrado = df_export[mask]

                    if df_filtrado.empty:
                        st.warning("No hay datos en ese rango de fechas.")
                    else:
                        import io
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            df_filtrado.to_excel(
                                writer, index=False, sheet_name="Historial")

                            total = len(df_filtrado)
                            total_ok = len(
                                df_filtrado[df_filtrado["Resultado"].str.contains("OK - Sano", na=False)])
                            total_nok = len(
                                df_filtrado[df_filtrado["Resultado"].str.contains("NOK", na=False)])

                            resumen = pd.DataFrame({
                                "Concepto": [
                                    "Máquina", "Tipo", "Periodo",
                                    "Total lecturas", "Lecturas OK",
                                    "Lecturas NOK", "% Disponibilidad"
                                ],
                                "Valor": [
                                    nombre, "Rodamiento / Torno CNC",
                                    f"{fecha_inicio} → {fecha_fin}",
                                    total, total_ok, total_nok,
                                    f"{round(total_ok / total * 100, 1)}%" if total > 0 else "—"
                                ]
                            })
                            resumen.to_excel(
                                writer, index=False, sheet_name="Resumen")

                        buffer.seek(0)
                        nombre_archivo = f"AuraPredict_{nombre}_{fecha_inicio}_{fecha_fin}.xlsx"
                        st.download_button(
                            label="⬇️ Descargar Excel",
                            data=buffer,
                            file_name=nombre_archivo,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        st.success(
                            f"✅ {len(df_filtrado)} lecturas entre "
                            f"{fecha_inicio} y {fecha_fin}. "
                            f"Disponibilidad: {round(total_ok / total * 100, 1) if total > 0 else 0}%"
                        )
    else:
        historial = obtener_historial_prensa(nombre, limite=50)
        if not historial:
            st.info("Aún no hay lecturas para esta máquina.")
        else:
            df = pd.DataFrame(historial, columns=[
                "Timestamp", "Desviacion_uE", "Vibracion_dB",
                "Particulas_ISO", "Resultado"])
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
            # --- EXPORTACIÓN EXCEL PRENSA ---
            st.markdown("---")
            st.subheader("📥 Exportar Historial")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                fecha_inicio = st.date_input(
                    "Desde",
                    value=pd.Timestamp.now() - pd.Timedelta(days=30),
                    key="fecha_ini_prensa"
                )
            with col_f2:
                fecha_fin = st.date_input(
                    "Hasta",
                    value=pd.Timestamp.now(),
                    key="fecha_fin_prensa"
                )

            if st.button("Generar Excel", key="excel_prensa", use_container_width=True):
                historial_completo = obtener_historial_prensa(
                    nombre, limite=10000)
                if not historial_completo:
                    st.warning("No hay datos para exportar.")
                else:
                    df_export = pd.DataFrame(
                        historial_completo,
                        columns=["Timestamp", "Desviacion_uE",
                                 "Vibracion_dB", "Particulas_ISO", "Resultado"]
                    )
                    df_export["Timestamp"] = pd.to_datetime(
                        df_export["Timestamp"])
                    mask = (
                        (df_export["Timestamp"].dt.date >= fecha_inicio) &
                        (df_export["Timestamp"].dt.date <= fecha_fin)
                    )
                    df_filtrado = df_export[mask]

                    if df_filtrado.empty:
                        st.warning("No hay datos en ese rango de fechas.")
                    else:
                        import io
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                            df_filtrado.to_excel(
                                writer, index=False, sheet_name="Historial")

                            total = len(df_filtrado)
                            total_ok = len(
                                df_filtrado[df_filtrado["Resultado"].str.contains("OK", na=False)])
                            total_nok = len(
                                df_filtrado[df_filtrado["Resultado"].str.contains("NOK", na=False)])

                            resumen = pd.DataFrame({
                                "Concepto": [
                                    "Máquina", "Tipo", "Periodo",
                                    "Total lecturas", "Lecturas OK",
                                    "Lecturas NOK", "% Disponibilidad"
                                ],
                                "Valor": [
                                    nombre, "Prensa Hidráulica",
                                    f"{fecha_inicio} → {fecha_fin}",
                                    total, total_ok, total_nok,
                                    f"{round(total_ok / total * 100, 1)}%" if total > 0 else "—"
                                ]
                            })
                            resumen.to_excel(
                                writer, index=False, sheet_name="Resumen")

                        buffer.seek(0)
                        nombre_archivo = f"AuraPredict_{nombre}_{fecha_inicio}_{fecha_fin}.xlsx"
                        st.download_button(
                            label="⬇️ Descargar Excel",
                            data=buffer,
                            file_name=nombre_archivo,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        st.success(
                            f"✅ {len(df_filtrado)} lecturas entre "
                            f"{fecha_inicio} y {fecha_fin}. "
                            f"Disponibilidad: {round(total_ok / total * 100, 1) if total > 0 else 0}%"
                        )
