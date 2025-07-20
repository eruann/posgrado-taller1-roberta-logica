# Técnicas de Data Science Aplicadas en los Experimentos

## **1. Procesamiento de Embeddings**
- **Extracción de embeddings** de capas específicas del modelo RoBERTa-base
- **Concatenación de vectores** (premisa + hipótesis + diferencia)
- **Cálculo de embeddings delta** (diferencia elemento a elemento)

## **2. Técnicas de Normalización**
- **All-but-the-mean (ABTM)**: Eliminación de la media global
- **Per-type normalization**: Normalización separada por tipo de vector
- **Standard scaling**: Estandarización (media=0, std=1)
- **L2 normalization**: Normalización de vectores unitarios
- **Sin normalización**: Datos crudos

## **3. Reducción de Dimensionalidad**
- **PCA (Principal Component Analysis)**: Reducción a ~50 dimensiones
- **ZCA Whitening**: Blanqueado de componentes principales
- **UMAP**: Reducción a 2D para visualización
- **Incremental PCA**: Para datasets grandes con procesamiento por lotes

## **4. Técnicas de Deflación**
- **Top-K PC removal**: Eliminación de las primeras K componentes principales
- **Deflación por lotes**: Procesamiento eficiente en GPU

## **5. Clustering y Análisis No Supervisado**
- **K-Means clustering** (k=3) para agrupar proyecciones 2D
- **Evaluación de pureza de clusters**
- **NMI (Normalized Mutual Information)** para evaluación

## **6. Análisis Estadístico**
- **Bootstrap sampling** para intervalos de confianza
- **Permutation tests** para significancia estadística
- **Chi-square tests** para independencia
- **Mann-Whitney U tests** para comparación de distribuciones
- **Kolmogorov-Smirnov tests** para diferencias de distribución
- **Cohen's d** para tamaño del efecto
- **Corrección de Benjamini-Hochberg** para múltiples comparaciones

## **7. Análisis de Anisotropía**
- **S_intra**: Anisotropía intra-cluster
- **S_inter**: Anisotropía inter-cluster
- **Análisis contrastivo** con diferencias cruzadas

## **8. Machine Learning Supervisado**
- **Árboles de decisión** (profundidad máxima ~4)
- **Clasificadores lineales** para detectar cuantificadores (∀, ∃)
- **Probing de dimensiones individuales**

## **9. Técnicas de Evaluación**
- **Cross-validation** para validación de modelos
- **Matrices de confusión** para evaluación de clasificación
- **Adjusted Rand Index** para evaluación de clustering

## **10. Optimización y Procesamiento**
- **Procesamiento por lotes (batching)** para datasets grandes
- **GPU acceleration** con cuML y CuPy
- **Memory management** agresivo para datasets grandes
- **Incremental processing** para evitar OOM

## **11. Frameworks y Librerías Específicas**
- **PyTorch + Hugging Face** para modelos de lenguaje
- **cuML** para algoritmos de ML en GPU
- **UMAP-learn** para reducción de dimensionalidad
- **scikit-learn** para algoritmos de ML clásicos
- **MLflow** para experiment tracking
- **Pandas + PyArrow** para manejo de datos

## **12. Análisis Comparativo**
- **Comparación SNLI vs FOLIO** para análisis de "geometría lógica"
- **Análisis de diferencias entre datasets**
- **Evaluación de robustez** entre diferentes configuraciones

---

*Estas técnicas se aplicaron sistemáticamente para investigar si los embeddings capturan implícitamente reglas de inferencia lógica de primer orden, comparando diferentes estrategias de normalización y análisis.* 