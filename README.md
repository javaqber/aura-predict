# 🔮 AuraPredict: Sistema Avanzado de Mantenimiento Predictivo Industrial

AuraPredict es una plataforma integral de **Mantenimiento Predictivo (PdM)** diseñada para entornos de Industria 4.0. Utiliza Inteligencia Artificial (Machine Learning No Supervisado) y telemetría avanzada para monitorizar la salud de activos industriales críticos en tiempo real.

El objetivo principal es predecir fallos catastróficos y desgaste de componentes antes de que provoquen paradas de producción no planificadas, permitiendo la transición del mantenimiento preventivo al predictivo.

## 🏗️ Arquitectura de Microservicios

El proyecto está diseñado bajo una arquitectura desacoplada, lista para despliegues en _Edge Computing_ o _Cloud_:

1. **AI Core (Machine Learning):**
   - Implementación de algoritmos de detección de anomalías (`Isolation Forest`) para modelar la curva P-F (Potential to Failure).
   - Capaz de aprender la "firma de salud" base de una máquina e identificar desviaciones sutiles.
2. **Backend Engine (FastAPI):**
   - API REST de alto rendimiento que centraliza la lógica de diagnóstico.
   - Valida estrictamente los payloads de telemetría provenientes de los PLCs o sensores IoT de la planta.
3. **Frontend Dashboard (Streamlit):**
   - Interfaz de usuario (HMI) diseñada para jefes de planta y directores de mantenimiento.
   - Visualización clara de alertas, estados y recomendaciones operativas en tiempo real.

## 🏭 Activos Monitorizados (Casos de Uso)

AuraPredict soporta múltiples perfiles de maquinaria mediante endpoints dedicados:

- **⚙️ Tornos CNC y Fresadoras (Mecanizado):**
  - Analiza telemetría de vibración de alta frecuencia (RMS, Peak-to-Peak, Kurtosis, Skewness).
  - Detecta micro-grietas en rodamientos y desgaste de husillos días antes de la rotura.
- **🏗️ Prensas Hidráulicas de Extrusión (Aluminio):**
  - Evalúa el riesgo de rotura por fatiga estructural leyendo micro-deformaciones ($\mu\epsilon$) en las columnas (Tie Rods).
  - Monitoriza la firma acústica (dB) y la contaminación del aceite (ISO) para predecir fallos en el bloque hidráulico principal.

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

4. Ejecutar los servicios (Requiere dos terminales):

- **Terminal 1(API REST)**

```bash
    python -m uvicorn src.api:app --reload
```

- **Terminal 2(Dashboard Web)**

```bash
    python -m streamlit run src/dashboard.py
```

## 📡 Ejemplo de Interacción API

Los sensores en planta envían peticiones POST automáticas al servidor.

**Prensa de Extrusión: POST /predict/extrusion_press**

```json
{
  "Desviacion_Columnas_uE": 2.5,
  "Vibracion_Bomba_AltaFrec_dB": 65.0,
  "Particulas_Aceite_ISO": 14
}
```

**Respuesta de la IA**

```json
{
  "Desviacion_Columnas_uE": 2.5,
  "Vibracion_Bomba_AltaFrec_dB": 65.0,
  "Particulas_Aceite_ISO": 14
}
```
