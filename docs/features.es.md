# Características

## El dataset

- **8 especies de mamíferos** nativas del Parque Nacional de Doñana (ciervo rojo,
  gamo, jabalí, lince ibérico, zorro rojo, meloncillo, conejo europeo, tejón) — ver
  [Clases](dataset-description.md#clases).
- **Anotaciones en formato YOLO**: un fichero de etiquetas por imagen, cuadros
  delimitadores normalizados, listos para entrenar un modelo de detección de objetos
  sin ningún paso de conversión.
- **Particiones `train`/`val`/`test` estratificadas** (70/20/10 por defecto),
  construidas para que cada especie esté representada en cada partición — ver
  [Particiones del dataset](dataset-description.md#particiones-del-dataset).
- **Almacenamiento en shards `.tar`** en HuggingFace Hub, manteniendo manejable el
  número de objetos individuales del repositorio a gran escala, mientras sigue
  reconstruyendo una estructura simple `images/<split>/` + `labels/<split>/` al
  extraerlos.
- **Licencia abierta** — CC BY 4.0, con metadatos de cita (`CITATION.cff`) generados y
  mantenidos sincronizados automáticamente.
- **Publicado en múltiples repositorios**, cada uno con su propio DOI/PID cuando
  aplica (HuggingFace Hub, Zenodo, B2SHARE, GBIF) — consulta la
  [Guía de publicación](publishing-guide.md) para ver la lista completa y qué se
  almacena dónde.
- **Soporte para Camtrap DP** en GBIF: las detecciones se exponen como un paquete
  [Camtrap DP](https://camtrap-dp.tdwg.org/), haciendo que el dataset sea descubrible
  por ecólogos e investigadores en conservación, no solo por la comunidad de ML.

## La CLI de `donadataset`

- **`donadataset generate real`** — construye el dataset YOLO limpio y versionado a
  partir de los datos fuente en bruto (particionado, muestreo estratificado,
  validación de clases).
- **`donadataset generate toy`** — extrae un pequeño subconjunto de un dataset ya
  generado, útil para pruebas locales rápidas.
- **Un comando de publicación por repositorio** — `publish huggingface`,
  `publish zenodo`, `publish b2share`, `publish gbif` — cada uno con sus propios pasos
  `prepare`/`upload`/`release` (o equivalentes), para que cada plataforma se pueda
  gestionar de forma independiente.
- **`wizard` interactivo** para HuggingFace Hub y Zenodo — recorre cada fase paso a
  paso, pide confirmación antes de acciones irreversibles (hacer público un dataset,
  publicar un registro de Zenodo), detecta y retoma ejecuciones parcialmente
  completadas, y te deja reintentar, saltar o abortar un paso fallido en vez de fallar
  en seco.
- **`pipeline` no interactivo** por repositorio, para ejecuciones scriptadas/
  automatizadas de la misma secuencia sin ningún prompt (usado por CI o
  `publish all`).
- **`donadataset publish all`** — orquesta HuggingFace Hub → Zenodo → B2SHARE → GBIF
  de principio a fin en un único comando, cerrando automáticamente los huecos
  manuales entre ellos (p. ej. volver a subir a HuggingFace tras reservar Zenodo un
  DOI). Soporta `--include`/`--exclude` para seleccionar repositorios y `--dry-run`
  para previsualizar el plan.
- **Gestión de configuración integrada** — `config show`/`config set`/`config
  wizard` (global y por integración) leen y escriben `settings.toml`, con los tokens de
  acceso guardados como ajustes reales pero siempre enmascarados en `show` e
  introducidos mediante entrada oculta en `set`/`wizard`, nunca mostrados en la
  terminal.
- **Soporte de `--dry-run`** en prácticamente todos los comandos de publicación, para
  previsualizar exactamente qué ocurriría sin tocar ninguna API remota.
- **Exportaciones autoverificadas** — cada paso `prepare` recalcula y comprueba
  checksums contra su propia salida antes de subir nada, y los comandos
  `download`/`sync-*` verifican el viaje de ida y vuelta tras publicar.
- **Herramientas de documentación** (`cli.py docs build`/`serve`/`pdf`) — construyen
  este sitio MkDocs, lo sirven en local para editarlo, o lo renderizan a un único PDF.
