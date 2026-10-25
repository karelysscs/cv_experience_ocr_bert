# Estimador de años de experiencia en CVs — PaddleOCR + BERT

> **Resultado principal:** en 600 CVs de prueba, BERT ajustado estima los años de experiencia con un
> error medio de **0.23 años** (99.7 % de los CVs dentro de ±2 años), frente a 0.81 de TF-IDF y 3.50 de
> la mediana. Una regla que solo suma fechas queda casi igual (0.26).

## 1. ¿Qué hace el proyecto y para quién?

El sistema lee la **imagen de un CV**, extrae su texto con **PaddleOCR** y predice los **años de
experiencia laboral (0–20)** del candidato con un **regresor BERT ajustado** (*fine-tuned*). Se compara
con tres modelos más simples (mediana, TF-IDF + Ridge y *embeddings* de oraciones + Ridge) y con una
regla extra que suma fechas, todos sobre la misma partición de datos.

Usuarios posibles: agencias de reclutamiento, bolsas de trabajo, servicios públicos de empleo y equipos
de RR. HH. que reciben miles de CVs y quieren **prellenar el perfil del candidato** u **ordenar qué CVs
leer primero**. Nunca debe decidir a quién se entrevista o contrata (ver §11).

## 2. Estructura del repositorio

```
app.py                                  App de Streamlit (4 pestañas)
notebooks/00_colab_full_pipeline.ipynb  Colab: todo el pipeline en un solo notebook
notebooks/01_ocr_gpu.ipynb              Kaggle/Colab: muestreo + PaddleOCR (GPU) + limpieza + split
notebooks/02_models_gpu.ipynb           Kaggle/Colab: baselines + BERT (GPU) + evaluación
src/download_data.py       muestreo estratificado de PopResume (streaming)
src/ocr_extraction.py      PaddleOCR, un JSON por CV, reanudable
src/ocr_quality.py         CER / WER / años y rangos de fechas bien leídos
src/preprocessing.py       orden de lectura, limpieza, datos personales, enmascarado
src/split.py               partición estratificada 70/15/15
src/baseline_median.py     Baseline 0
src/baseline_tfidf.py      Baseline 1
src/baseline_embeddings.py Baseline 2
src/baseline_rules.py      extra: aritmética de fechas
src/train_bert.py          ajuste de BERT como regresor
src/probe.py               prueba de la P1 (mover fechas / borrar un trabajo)
src/evaluation.py          métricas, intervalos bootstrap, tablas de P2–P4
src/visualization.py       figuras
src/utils.py               rutas, semilla, constantes
```

`data/raw/images/` y `models/` no se suben a GitHub (`.gitignore`); el §12 explica cómo regenerarlos.

## 3. Muestreo y variable objetivo

* Dataset: [PopResume](https://huggingface.co/datasets/sumin-yu/PopResume) (Yu, Park & Moon, EMNLP 2026, CC BY 4.0).
* **800 CVs por ocupación × 5 = 4,000**, solo imágenes sin foto (`has_photo == False`), leídas por
  *streaming* una ocupación a la vez y mezcladas con **semilla 42** (`buffer_size=5000`).
  IDs en `data/raw/sampled_ids.txt`. Como el `id` de PopResume se repite entre ocupaciones, cada CV se
  identifica como `ocupación_id` (p. ej. `registered_nurses_00020`).
* Objetivo `exp_year` recortado a **0–20**. Solo **30 CVs (0.75 %)** llegan al tope, muy por debajo del
  15 %, así que el tope se mantiene en 20 (`output/metrics/split.json`).
* Distribución (`output/figures/experience_distribution.png`): sesgada a la derecha, media **6.5** y
  mediana **5.5** años; 18.6 % de los CVs en 0–2, 31.2 % en 3–5, 31.7 % en 6–10 y 18.5 % en 11–20 años.
  Los obreros de construcción tienen la mediana más baja (4.6) y los contadores la más alta (6.4).

## 4. OCR

* PaddleOCR **3.7.0** sobre PaddlePaddle **3.3.1** (versión GPU), modelo **`PP-OCRv5_server_det` +
  `PP-OCRv5_server_rec`**: el reconocedor *server* lee mejor los dígitos pequeños que el *mobile*, y las
  fechas son la evidencia que necesitan los modelos. Se desactivan la clasificación de orientación, el
  *unwarping* y la orientación de líneas, porque las imágenes son documentos digitales derechos.
* Por cada línea se guarda el texto, la confianza y la caja (`data/ocr/<id>.json`).
* **Orden de lectura** (`preprocessing.rebuild_reading_order`): las cajas se agrupan en filas por su
  posición vertical y cada fila se lee de izquierda a derecha. Un diseño **a dos columnas** se detecta
  cuando una posición horizontal (entre el 20 % y el 80 % del ancho) la cruza como máximo el 5 % de las
  cajas y cada lado tiene al menos el 15 %; entonces se lee el encabezado, luego la columna izquierda y
  después la derecha, para que un cargo nunca se separe de sus fechas. Los CVs de PopResume resultaron
  ser de una sola columna, así que en la práctica las filas se leen de arriba abajo.
* Hardware: **Google Colab, NVIDIA Tesla T4** · tiempo por CV: **0.89 s** (≈ 60 min para 4,000) ·
  confianza media por línea: **0.982** · imágenes fallidas o vacías: **0** (`output/metrics/ocr_summary.json`).

## 5. Calidad del OCR

Ambos textos (OCR y referencia) se normalizan igual: minúsculas, guiones unificados, puntuación
separada y espacios simples.

| Ocupación | CER | WER | Años bien leídos | Rangos de fechas bien leídos |
|---|---|---|---|---|
| Contadores y auditores | 0.38 % | 2.66 % | 98.0 % | 97.6 % |
| Obreros de construcción | 0.32 % | 2.23 % | 98.6 % | 95.5 % |
| Profesores de primaria / secundaria | 0.39 % | 2.69 % | 97.0 % | 95.5 % |
| Enfermeros | 0.42 % | 2.81 % | 98.4 % | 97.9 % |
| Desarrolladores de software | 0.48 % | 3.25 % | 98.0 % | 97.4 % |
| **Total (4,000 CVs)** | **0.40 %** | **2.73 %** | **98.0 %** | **96.8 %** |

El 88.4 % de los CVs tiene todos sus años bien leídos. Dónde falla: un dígito que se pierde en un año
(`2020.02` leído como `220.02`, `2017` como `207`) y el guion entre dos fechas; además, los nombres en
mayúsculas a veces se leen sin espacio (`TYLER VANCE` → `TYLERVANCE`), lo que se resuelve en la
limpieza (§6). Los CVs con mayor CER están en `worst_10_ids_by_cer` (`output/metrics/ocr_quality.json`).

## 6. Limpieza y enmascarado

* Se unen líneas cortadas y palabras separadas con guion; se normalizan guiones y espacios.
* Datos personales eliminados: nombre (la primera línea solo con letras del encabezado, aunque el OCR
  haya pegado nombre y apellido), correo, teléfono, URLs, dirección, "Ciudad, ST ZIP" y la línea de
  estado "AL, USA". Una primera versión no ocultaba los nombres leídos sin espacio en 363 CVs (9 %);
  se detectó al revisar los resultados y todos los modelos se reentrenaron con la corrección.
* Líneas de baja confianza: se conservan. La distribución de confianza hace inútil un umbral: el 1 % de
  las líneas tiene menos de 0.951 y **ninguna** baja de 0.80, así que la variante `ocr_conf` es igual al
  texto limpio.
* **Enmascarado**: después de leer 50 textos de OCR al azar (`output/tables/inspection_sample.txt`), las
  frases que dicen la experiencia total ("12 years of experience", "Experience: 7 yrs", "a decade of
  experience"), la edad o la fecha de nacimiento se reemplazan por `[MASK]`. **0 de 4,000 CVs** las
  tenían: los CVs de PopResume solo tienen encabezado, *Work History* con fechas (`2018.08-2021.06`) y
  *Education* (`output/metrics/masking.json`). Las fechas de trabajo y de graduación se conservan.

## 7. Partición

70 / 15 / 15 por CV, estratificada por ocupación × banda de experiencia (0–2, 3–5, 6–10, 11–20),
semilla 42: **2,800 / 600 / 600 CVs**, con la misma proporción de bandas en cada parte (±0.3 puntos).
Archivos en `data/splits/`; todos los modelos los usan. Todo el ajuste se hace en validación; el test
se usa una sola vez.

## 8. Baselines

| Modelo | Configuración |
|---|---|
| Mediana | siempre 5.52 años (mediana del entrenamiento) |
| TF-IDF + Ridge | n-gramas de palabras 1–2 + de caracteres 3–5 (elegidos en validación), TF sublineal, **los años se mantienen como tokens** (son evidencia; la P3 prueba quitarlos); α = 0.1 elegido en validación |
| Embeddings + Ridge | `BAAI/bge-small-en-v1.5` congelado; un CV de más de 510 tokens se dividiría en fragmentos solapados (paso 64) y se promediaría, pero todos los CVs caben en un fragmento; α = 0.1 elegido en validación |
| Regla de fechas (extra) | rangos de fechas por regex (`AAAA.MM-AAAA.MM`, nombres de mes, `MM/AAAA`) dentro de *Work History*, se unen traslapes y se suman duraciones; ningún CV usa "Present"; si no lee ningún rango usa la mediana (0.8 % del test) |

## 9. BERT

* `distilbert-base-uncased`, `AutoModelForSequenceClassification(num_labels=1, problem_type="regression")`.
* Objetivo escalado a [0, 1] (÷ 20); las predicciones se vuelven a años y se recortan a 0–20. Pérdida MSE.
* Textos largos: truncado por la cabeza. Los CVs son cortos (mediana de 108 tokens, percentil 90 de 149),
  así que **ningún CV supera el límite y el 100 % de los años se conserva** (`date_survival` en
  `output/metrics/runs.json`).
* lr 5e-5, batch 16, hasta 10 épocas, *warm-up* del 10 %, *weight decay* 0.01, *early stopping*
  (paciencia 2) sobre el MAE de validación, semilla 42, longitud máxima 256.
* Hardware: la primera corrida se hizo en una GPU T4 de Colab (≈ 1.5 min). Tras corregir el enmascarado
  de nombres se agotó la cuota gratuita de GPU de Colab, así que el modelo final se entrenó en la **CPU
  de una laptop** (Intel, 10 hilos, ≈ 30 min por época). El modelo principal se **detuvo a mano después
  de la época 6** (190 min) y se usó ese checkpoint, que era el mejor: el MAE de validación bajó en cada
  época (0.82 → 0.78 → 0.54 → 0.45 → 0.33 → **0.24**), así que con más épocas podría mejorar un poco más.
  La variante de la P3 (OCR crudo) se detuvo sola por *early stopping* en la época 5 (141 min).

## 10. Resultados (test, 600 CVs)

`output/tables/model_comparison.csv`

| Modelo | MAE (años) | RMSE | Acc ±1 | Acc ±2 | Acc ±3 | Acc. bandas | Macro-F1 bandas | R² | Entrenamiento | Inferencia / CV |
|---|---|---|---|---|---|---|---|---|---|---|
| Mediana | 3.50 | 4.68 | 17.7 % | 38.5 % | 55.3 % | 31.8 % | 0.12 | −0.05 | < 1 s | < 0.1 ms |
| TF-IDF + Ridge | 0.81 | 1.10 | 70.0 % | 93.8 % | 98.3 % | 86.5 % | 0.87 | 0.94 | 4 s | 1.4 ms |
| Embeddings + Ridge | 1.12 | 1.57 | 57.7 % | 86.3 % | 93.3 % | 78.5 % | 0.79 | 0.88 | 4.6 min | 60 ms |
| Regla de fechas (extra) | 0.26 | 0.65 | 94.3 % | 95.7 % | 99.0 % | 94.3 % | 0.95 | 0.98 | 7 s | 0.2 ms |
| **BERT ajustado** | **0.23** | **0.33** | **98.3 %** | **99.7 %** | **100 %** | **95.5 %** | **0.96** | **0.995** | 190 min (CPU) | 261 ms (CPU) |

**Ganancia de BERT en MAE** (bootstrap pareado al 95 %, 2,000 remuestreos, `output/tables/bert_gain_ci.csv`):

| Frente a | Años de error que BERT ahorra por CV | IC 95 % | ¿Significativa? |
|---|---|---|---|
| Mediana | 3.28 | [3.03, 3.53] | Sí |
| TF-IDF + Ridge | 0.58 | [0.52, 0.64] | Sí |
| Embeddings + Ridge | 0.89 | [0.80, 0.98] | Sí |
| Regla de fechas | 0.03 | [−0.01, 0.09] | **No** |

Figuras: `pred_vs_true.png`, `mae_by_band.png`, `mae_by_occupation.png`, `confusion_*.png`, `bert_gain_ci.png`.

**Los 10 peores errores de BERT** (`output/tables/bert_top10_errors.csv`, columna `cause`): todos son
sobreestimaciones de 1.0 a 2.3 años. Causas, comparando cada CV con la suma exacta de sus fechas:
5 por **trabajos que se traslapan** (BERT suma dos veces los meses compartidos), 4 por **vacíos
laborales** entre trabajos (BERT los cuenta como experiencia) y 1 por **años mal leídos por el OCR**
(7 de 9 años correctos). 8 de los 10 son CVs de 16 años o más con 3–4 trabajos.

Posibles mejoras: entrenar más épocas en GPU (el MAE de validación seguía bajando), agregar como
variable la suma de fechas que calcula la regla (un modelo híbrido) y aumentar los ejemplos con
traslapes y vacíos.

## 11. Preguntas analíticas

**P1 — ¿Qué aprende BERT que los baselines no?** BERT reduce el MAE en 0.58 años frente a TF-IDF y en
0.89 frente a los *embeddings*, ganancias estadísticamente significativas. La ganancia viene sobre todo
de los extremos: en los CVs de 11–20 años TF-IDF se equivoca 1.28 años y los subestima en 0.81 (su
recta predicho–real tiene pendiente 0.90: "tira al centro"), mientras que BERT se equivoca 0.32 años con
pendiente 1.01; en los juniors (0–2) baja de 0.66 a 0.27. La prueba (`output/tables/probe_summary.csv`,
20 CVs del test) muestra qué usa cada modelo:
* **Borrar un trabajo** debería bajar la experiencia en 1.25 años en promedio. BERT baja 1.15 y se mueve
  en la dirección correcta en el 75 % de los CVs; TF-IDF y los *embeddings* suben (+0.48 y +0.15):
  no conectan el cargo con sus fechas.
* **Mover todas las fechas +5 años** no cambia ninguna duración, así que la predicción no debería
  cambiar. La regla no cambia, pero BERT baja 1.56 años en promedio: además de las duraciones, usa el
  valor absoluto de los años como pista, no solo las diferencias entre ellos.

Conclusión: BERT sí aprende la estructura "cargo + fechas" que los baselines no ven, pero mezcla esa
aritmética con atajos de vocabulario sobre los años. La regla de fechas, que solo hace aritmética, queda
a 0.03 años de BERT (diferencia no significativa).

**P2 — ¿Cuánto cuesta el ruido del OCR?** El OCR es muy bueno (CER 0.4 %, WER 2.7 %, 98 % de los años
bien leídos), así que el costo es pequeño para los modelos entrenados con el propio texto del OCR
(`output/tables/ocr_vs_reference.csv`): TF-IDF +0.004 años, *embeddings* −0.02 y BERT −0.04 (BERT
incluso funciona un poco mejor con el texto del OCR, con el que se entrenó, que con el texto original,
que tiene otro formato de viñetas). El más sensible es la **regla de fechas**: 0.12 años de MAE con el
texto original frente a 0.26 con el OCR, porque un dígito mal leído rompe un rango de fechas completo.
Los modelos aprendidos son más robustos porque vieron el ruido del OCR durante el entrenamiento.

**P3 — Sensibilidad metodológica** (`output/tables/sensitivity.csv`).
* *Limpieza del OCR* — OCR crudo (sin orden de lectura, sin unir líneas y sin ocultar datos personales)
  frente a OCR limpio, en todos los modelos: BERT empeora +0.11 (0.23 → 0.34; la variante cruda paró sola
  en la época 5), los *embeddings* +0.48, TF-IDF +0.10 y la regla no cambia (sus expresiones regulares no
  dependen del formato). La limpieza ayuda a todos los modelos aprendidos. El filtro de líneas de baja
  confianza no se pudo comparar porque ninguna línea baja de 0.80.
* *Números en TF-IDF* — quitar los dígitos sube el MAE de 0.81 a 1.77: los años son la evidencia
  principal, como se esperaba.

**P4 — Valor de negocio, equidad y ética.** Un estimador así sirve para **prellenar el perfil** de un
candidato en una bolsa de trabajo o para **ordenar** miles de CVs antes de que un reclutador los lea;
no para decidir a quién se entrevista. Un error de ±2 años casi no importa para prellenar un perfil
(BERT acierta dentro de ±2 años en el 99.7 % de los casos), pero ±5 años podría cambiar el nivel del
puesto. Subestimar es peor para el candidato (podría quedar fuera de un filtro), y sobreestimar es peor
para el empleador; BERT tiende a **sobreestimar levemente** (+0.08 años en promedio, +0.23 en los CVs de
11–20 años).

*Equidad* (`output/tables/fairness.csv`, BERT en el test):

| Grupo | n | MAE (años) | Error medio (predicho − real) |
|---|---|---|---|
| Mujeres | 293 | 0.23 | +0.08 |
| Hombres | 307 | 0.22 | +0.07 |
| Asiático / isleño del Pacífico | 118 | 0.22 | +0.08 |
| Negro | 37 | 0.18 | +0.01 |
| Blanco | 445 | 0.23 | +0.08 |

No hay un sesgo sistemático por sexo ni por raza: las diferencias son de centésimas de año y el grupo más
pequeño (37 CVs) tiene un intervalo amplio. Aunque sexo y raza no son variables del modelo, podrían
filtrarse a través de la ocupación o de los cargos, por eso se revisaron.

*¿Vale la pena BERT?* Frente a TF-IDF, sí: reduce el error en 0.58 años de forma significativa. Frente a
la regla de fechas, no en este dataset: la regla es casi igual de precisa, no necesita entrenamiento y es
1,000 veces más rápida. La ventaja de BERT aparecería con CVs reales de formatos variados, donde una
expresión regular fija dejaría de funcionar.

*Limitaciones*: CVs sintéticos de una sola plantilla, 5 ocupaciones, estadísticas de EE. UU. y solo en
inglés. Por todo esto, el sistema nunca debe decidir a quién se contrata: solo debe apoyar, con revisión
humana.

## 12. Problemas encontrados y cómo se resolvieron

| Problema | Solución |
|---|---|
| La laptop no tiene una GPU compatible (Intel Iris Xe) | OCR y BERT en **Google Colab** con GPU T4 gratuita, en un solo notebook |
| PopResume es un dataset con acceso restringido | Aceptar las condiciones en Hugging Face y usar un token como secreto de Colab |
| Colab se quedó sin RAM al muestrear (5,000 imágenes descomprimidas en memoria) | Mantener las imágenes comprimidas (JPEG) mientras se mezclan |
| El `id` se repite entre ocupaciones | Identificar cada CV como `ocupación_id` |
| Fechas en formato `2018.08-2021.06` | Ampliar las expresiones regulares de fechas |
| Nombres leídos sin espacio (9 % de los CVs) | Nueva regla de nombres y reentrenamiento de todos los modelos |
| Se agotó la cuota de GPU de Colab | Reentrenar BERT en la CPU de la laptop (más lento, mismos resultados reproducibles) |

Lección principal: probar el pipeline completo con una muestra pequeña (unos 50 CVs) antes de procesar
los 4,000.

## 13. Cómo reproducirlo

### Opción A (recomendada) — todo en un notebook de Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karelysscs/cv_experience_ocr_bert/blob/Practica_04/notebooks/00_colab_full_pipeline.ipynb)

`notebooks/00_colab_full_pipeline.ipynb` corre todo el pipeline en una GPU T4 con *Ejecutar todas*:
muestreo, PaddleOCR, calidad del OCR, limpieza, partición, todos los modelos, las preguntas P1–P4, las
figuras y la app de Streamlit con un enlace público, y sube los resultados a GitHub. Necesita los
secretos de Colab `HF_TOKEN` (Hugging Face) y `GH_TOKEN` (token *fine-grained* de GitHub con permiso de
lectura y escritura de contenidos solo en este repositorio). El avance se guarda en Google Drive, así
que si Colab se desconecta, se retoma donde quedó. PaddlePaddle-GPU corre en un entorno virtual aislado
porque choca con las librerías CUDA de PyTorch.

### Opción B — notebooks de Kaggle + computadora local

Las etapas con GPU corren en **Kaggle o Google Colab**; el resto, en local.

1. **Acceso:** aceptar las condiciones del dataset en Hugging Face, crear un token de lectura y guardarlo
   como secreto `HF_TOKEN` en Kaggle (*Add-ons → Secrets*) o Colab (*🔑 Secretos*).
2. **Etapa 1 (GPU):** abrir `notebooks/01_ocr_gpu.ipynb` en Kaggle/Colab, activar la GPU y ejecutar todo.
   Muestrea los CVs, corre PaddleOCR, mide la calidad, limpia el texto y crea la partición. Descargar
   `stage1_ocr_results.zip`, descomprimirlo en el repo, hacer commit y push.
3. **Etapa 2 (GPU):** abrir `notebooks/02_models_gpu.ipynb` y ejecutar todo. Entrena los modelos y las
   variantes de la P3, corre la prueba, la evaluación y las figuras. Descargar `stage2_outputs.zip`
   (se sube al repo) y `stage2_models.zip` (se descomprime en local; no se sube).

Comandos equivalentes (desde la raíz del repo):

```bash
python -m src.download_data --n-per-occupation 800
python -m src.ocr_extraction --model server --device gpu
python -m src.ocr_quality
python -m src.preprocessing --inspect 50   # leer primero la muestra
python -m src.preprocessing --min-conf 0.80
python -m src.split
python -m src.baseline_median
python -m src.baseline_tfidf
python -m src.baseline_embeddings
python -m src.baseline_rules
python -m src.train_bert --max-length 256
python -m src.probe
python -m src.evaluation
python -m src.visualization
```

## 14. Cómo correr la app de Streamlit

Usar Python **3.10–3.13** (PaddlePaddle aún no tiene versión para 3.14):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate en macOS/Linux)
pip install -r requirements.txt
streamlit run app.py
```

Las pestañas 1–3 solo necesitan los archivos de `output/` y `data/` del repo. La pestaña 4 (estimador en
vivo) corre PaddleOCR *mobile* en CPU y necesita los modelos entrenados en `models/`. La forma más simple
de mostrarla es la última celda del notebook de Colab, que la abre con un enlace público.

## 15. Flujo de trabajo en Git

El trabajo se hizo en ramas *feature* (`feature/data-sampling`, `feature/paddleocr-extraction`,
`feature/baselines`, `feature/bert-regression`, `feature/evaluation`, `feature/streamlit-dashboard`, …) y
correcciones (`fix/...`), unidas por Pull Request a **`Practica_04`**, que al final se une a `main`.

## Video

Enlace en `video/link.txt`.
