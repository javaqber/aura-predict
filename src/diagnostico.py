def diagnosticar_rodamiento(rms, peak_to_peak, kurtosis, skewness):
    """
    Base de conocimiento para diagnóstico de fallos en rodamientos.
    Correlaciona la firma de vibración con el tipo de fallo probable,
    las consecuencias y la acción recomendada.
    """

    es_anomalia = (rms > 0.15 or kurtosis > 2.5 or peak_to_peak > 1.5)

    if not es_anomalia:
        return {
            "tipo_fallo":           "Ningún fallo detectado",
            "componente_afectado":  "—",
            "descripcion":          "La máquina opera dentro de parámetros normales.",
            "consecuencias":        "—",
            "accion_recomendada":   "Continuar monitorización periódica.",
            "pieza_referencia":     "—",
            "ventana_actuacion":    "Sin urgencia",
            "nivel_urgencia":       "verde",
            "confianza":            "Alta"
        }

    # FALLO CRÍTICO INMINENTE
    if rms > 0.40 and kurtosis > 6.0:
        return {
            "tipo_fallo":           "Fallo catastrófico inminente",
            "componente_afectado":  "Rodamiento completo",
            "descripcion":          "Degradación avanzada en múltiples componentes. Niveles de vibración extremos en todos los ejes.",
            "consecuencias":        "Rotura del rodamiento en horas. Posibles daños al husillo, herramienta y pieza mecanizada.",
            "accion_recomendada":   "PARAR MÁQUINA INMEDIATAMENTE. Sustituir rodamiento antes de reiniciar.",
            "pieza_referencia":     "Consultar placa de máquina. Típico: FAG 6205-2RS / SKF 6205-2Z o equivalente.",
            "ventana_actuacion":    "Inmediato — menos de 8 horas",
            "nivel_urgencia":       "rojo",
            "confianza":            "Muy alta"
        }

    # DEFECTO EN ELEMENTO RODANTE (bola o rodillo)
    if kurtosis > 5.0 and skewness > 0.5 and rms > 0.20:
        return {
            "tipo_fallo":           "Defecto en elemento rodante",
            "componente_afectado":  "Bola o rodillo del rodamiento",
            "descripcion":          "Impactos de alta energía con distribución asimétrica. Grieta o descamación en uno o varios elementos rodantes.",
            "consecuencias":        "Progresión rápida del daño. En 2-4 semanas puede derivar en fallo de pista completa.",
            "accion_recomendada":   "Programar sustitución del rodamiento en los próximos 7-10 días. Aumentar frecuencia de monitorización a cada 2h.",
            "pieza_referencia":     "Rodamiento completo. Ref. típica según husillo: FAG 6206 / SKF 6206-2Z.",
            "ventana_actuacion":    "7-10 días",
            "nivel_urgencia":       "rojo",
            "confianza":            "Alta"
        }

    # DEFECTO EN PISTA INTERIOR
    if kurtosis > 4.0 and rms > 0.25 and skewness > 0.3:
        return {
            "tipo_fallo":           "Defecto en pista interior",
            "componente_afectado":  "Pista interior del rodamiento",
            "descripcion":          "Impactos modulados detectados. La pista interior presenta marcas de fatiga o descamación localizada.",
            "consecuencias":        "Sin intervención, el defecto se extiende a toda la pista en 3-6 semanas.",
            "accion_recomendada":   "Programar sustitución en próxima parada. Revisar condiciones de montaje y apriete del rodamiento.",
            "pieza_referencia":     "Rodamiento completo (no se sustituye solo la pista). Ref. típica: NSK 6205 / SKF 6205-2RS.",
            "ventana_actuacion":    "15-20 días",
            "nivel_urgencia":       "naranja",
            "confianza":            "Alta"
        }

    # DEFECTO INCIPIENTE EN PISTA EXTERIOR
    if kurtosis > 2.5 and rms > 0.15 and kurtosis <= 4.0:
        return {
            "tipo_fallo":           "Defecto incipiente en pista exterior",
            "componente_afectado":  "Pista exterior del rodamiento",
            "descripcion":          "Impactos periódicos de baja energía. Inicio de fatiga superficial en pista exterior. Fallo más habitual en rodamientos de husillo.",
            "consecuencias":        "Fallo progresivo en 4-8 semanas si no se actúa.",
            "accion_recomendada":   "Planificar sustitución en próxima parada de mantenimiento. Verificar estado y tipo de lubricación.",
            "pieza_referencia":     "Rodamiento completo. Consultar manual de máquina. Ref. típica: FAG 6205 / 6206.",
            "ventana_actuacion":    "20-30 días",
            "nivel_urgencia":       "naranja",
            "confianza":            "Media-alta"
        }

    # DESEQUILIBRIO O DESALINEACIÓN
    if rms > 0.20 and kurtosis < 2.0 and peak_to_peak > 1.5:
        return {
            "tipo_fallo":           "Desequilibrio o desalineación",
            "componente_afectado":  "Husillo / acoplamiento / portaherramienta",
            "descripcion":          "Vibración elevada sin impactos. Señal sinusoidal característica de desequilibrio dinámico o desalineación.",
            "consecuencias":        "Aceleración del desgaste de rodamientos y reducción de calidad superficial en el mecanizado.",
            "accion_recomendada":   "Verificar equilibrado de herramienta y portaherramienta. Comprobar alineación del husillo. No requiere parada urgente.",
            "pieza_referencia":     "No requiere cambio de pieza. Operación de reglaje y equilibrado.",
            "ventana_actuacion":    "30-45 días",
            "nivel_urgencia":       "amarillo",
            "confianza":            "Media"
        }

    # LUBRICACIÓN INSUFICIENTE O DEGRADADA
    if rms > 0.12 and kurtosis < 2.0 and peak_to_peak < 1.5:
        return {
            "tipo_fallo":           "Lubricación insuficiente o degradada",
            "componente_afectado":  "Rodamiento — capa de lubricante",
            "descripcion":          "Incremento moderado de vibración sin impactos. Compatible con grasa degradada o nivel de lubricante insuficiente.",
            "consecuencias":        "Sin lubricación adecuada el desgaste se acelera entre 5 y 10 veces. Puede derivar en fallo en pocas semanas.",
            "accion_recomendada":   "Reaplicar grasa según especificación del fabricante. Relubricar y monitorizar durante 48h.",
            "pieza_referencia":     "Grasa: SKF LGMT 2 / FAG Arcanol LOAD150 o equivalente. No requiere sustitución del rodamiento si se actúa pronto.",
            "ventana_actuacion":    "48-72 horas para relubricar",
            "nivel_urgencia":       "amarillo",
            "confianza":            "Media"
        }

    # ANOMALÍA NO CLASIFICADA
    return {
        "tipo_fallo":           "Anomalía detectada — patrón no clasificado",
        "componente_afectado":  "Por determinar",
        "descripcion":          "Los valores se desvían de la normalidad pero no coinciden con un patrón de fallo conocido.",
        "consecuencias":        "Requiere inspección manual para determinar la causa.",
        "accion_recomendada":   "Realizar inspección visual y auditiva. Aumentar frecuencia de monitorización a cada 2 horas.",
        "pieza_referencia":     "Por determinar tras inspección.",
        "ventana_actuacion":    "Inspección en las próximas 24 horas",
        "nivel_urgencia":       "naranja",
        "confianza":            "Baja"
    }
