# 🔮 AuraPredict: Sistema Avanzado de Mantenimiento Predictivo Industrial

AuraPredict es una plataforma integral de **Mantenimiento Predictivo (PdM)** diseñada para entornos de Industria 4.0. Utiliza Inteligencia Artificial (Machine Learning No Supervisado) y telemetría avanzada para monitorizar la salud de activos industriales críticos en tiempo real, con soporte multi-empresa y autenticación segura.

El objetivo principal es predecir fallos catastróficos y desgaste de componentes antes de que provoquen paradas de producción no planificadas, permitiendo la transición del mantenimiento preventivo al predictivo.

🚀 Producción: aurapredict-dashboard.onrender.com
⚙️ API: aurapredict-api.onrender.com

## 🏗️ Arquitectura de Microservicios

El proyecto está diseñado bajo una arquitectura desacoplada en cuatro capas:

**AI Core (Machine Learning):**

- Algoritmo Isolation Forest para detección de anomalías sin datos de fallo previos.
- Modela la "firma de salud" base de cada máquina e identifica desviaciones sutiles.
- Motor de diagnóstico basado en reglas expertas (diagnostico.py) con 7 tipos de fallo clasificados.

**Backend Engine (FastAPI):**

- API REST con autenticación JWT (tokens de 8 horas, renovación automática).
- Endpoints protegidos con HTTPBearer y Depends(get_usuario_actual).
- Sistema de alertas por email (Gmail SMTP SSL) con intervalo mínimo configurable.

**Frontend Dashboard (Streamlit):**

- Interfaz multi-rol: admin con vista global + panel de administración; cliente con vista filtrada.
- Splash screen en primera carga, login con sesión JWT, cierre de sesión seguro.
- Exportación a Excel con filtro por fechas y hoja de resumen de disponibilidad.

**Adquisición de Datos (Scheduler / Raspberry Pi):**

- Scheduler autónomo con horario laboral configurable y modo anomalía automático.
- Script de lectura directa del sensor ADXL345 vía I2C para Raspberry Pi.
- Autenticación JWT en el scheduler para envío seguro de lecturas a la API.

## 🏭 Activos Monitorizados (Casos de Uso)

**⚙️ Tornos CNC y Fresadoras (Mecanizado):**

- Telemetría de vibración de alta frecuencia: RMS, Peak-to-Peak, Kurtosis, Skewness.
- Detecta desgaste de rodamientos, desalineación de eje, holguras mecánicas y fallos de pista.

**🏗️ Prensas Hidráulicas de Extrusión (Aluminio):**

- Micro-deformaciones en columnas (µε), firma acústica de bomba (dB) y contaminación de aceite (ISO 4406).
- Detecta desalineación estructural y desgaste hidráulico.

## 🚀 Tecnologías Empleadas

- **Data Science & ML:** Python 3.12, Pandas, Scikit-Learn (`IsolationForest`), Joblib.
- **Backend API:** FastAPI, Uvicorn, Pydantic.
- **Frontend Web:** Streamlit, Requests, CSS inyectado.

## 🛠️ Instalación

### Despliegue Local

1. Clonar el repositorio:

```bash
git clone https://github.com/javaqber/aura-predict.git
cd aura-predict
```

2. Crear y activar el entorno virtual:

```bash
    python -m venv venv
  source venv/Scripts/activate  # Windows

# source venv/bin/activate    # Linux/Mac
```

3. Instalar dependencias:

```bash
    pip install -r requirements.txt
```

4. Ejecutar los servicios (Requiere tres terminales):

- **Terminal 1(API REST)**

```bash
    python uvicorn api:app --reload --port 8000 --app-dir src
```

- **Terminal 2(Dashboard Web)**

```bash
    python -m streamlit run src/dashboard.py
```

- **Terminal 3 — Simulador de sensores para pruebas (opcional)**

```bash
    python src/simulador.py
```

- **Terminal 4 — Scheduler de adquisición automática (opcional)**

```bash
    python src/scheduler.py
```

## 📊 Funcionalidades del Dashboard

- **Vista global de estado:** tabla en tiempo real con estado OK/NOK de todos los activos (solo admin).
- **Panel de administración:** gestión de empresas, usuarios y asignación de máquinas (solo admin).
- **Historial gráfico:** evolución temporal de RMS/Kurtosis (rodamiento) y µε/dB/ISO (prensa).
- **Exportación a Excel:** historial filtrado por fechas con hoja de resumen y % de disponibilidad.
- **Alertas por email:** notificación automática con diagnóstico completo al detectar anomalía.
- **Splash screen:** pantalla de carga visual en la primera inicialización del servicio.
- **Multi-empresa:** cada cliente ve únicamente sus propias máquinas y datos.
