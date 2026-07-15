# Publicar en GBIF

Esta guía explica cómo se convierte DonaDataset para publicarlo en **GBIF** usando la
CLI de `donadataset`. Está dirigida al **mantenedor del dataset**.

---

## 1. Qué es GBIF

[GBIF](https://www.gbif.org) (Global Biodiversity Information Facility) es el mayor
agregador mundial de acceso abierto de datos de biodiversidad. Indexa cientos de
millones de registros de ocurrencia de especies procedentes de instituciones de
investigación, museos de historia natural y proyectos de ciencia ciudadana, y asigna un
**DOI** permanente a cada dataset publicado. GBIF es la plataforma de referencia para
ecólogos, biólogos de conservación y responsables de políticas medioambientales.

## 2. En qué formato publicamos

DonaDataset se publica en GBIF como **[Camtrap DP](https://camtrap-dp.tdwg.org/)** — el
estándar TDWG/GBIF para datos de cámaras trampa, soportado de forma nativa por IPT v3+.
Es un Frictionless Data Package: un descriptor `datapackage.json` más tres tablas CSV —
`deployments.csv` (colocaciones de cámaras), `media.csv` (una fila por foto), y
`observations.csv` (una fila por detección). GBIF lo ingiere convirtiéndolo internamente
a registros de ocurrencia Darwin Core.

GBIF no almacena las imágenes en sí — esas viven en
[HuggingFace Hub](publishing-huggingface.md). Por defecto, `media.filePath` es una ruta
relativa (`images/<split>/<filename>`, igual que la estructura de la exportación de
HuggingFace), no una URL funcional. Pasa `--link-media-to-huggingface` para que sea en
su lugar una URL real y persistente — consulta la entrada de `media.csv` de la sección 4
para ver exactamente a qué apunta esa URL (no es una URL por foto).

## 3. Qué inventa `gbif prepare`, y por qué

Este pipeline no registra coordenadas GPS por cámara ni fechas de despliegue en ningún
sitio (ni en los nombres de fichero, ni en un manifest) — la única información de fecha
por imagen que podría tener es el EXIF. Así que `prepare` no te pide que rellenes nada a
mano; en su lugar hace suposiciones razonables, todas claramente señaladas en la salida:

- **Un despliegue por partición** (`train`, `val`, `test`) — tratar toda una partición
  como una única "colocación de cámara" obviamente no es literalmente cierto, pero es la
  única agrupación que tiene este pipeline. Cada una recibe un punto ilustrativo
  distinto dentro del Parque Nacional de Doñana (ver `SPLIT_DEPLOYMENT_COORDINATES` en
  `donadataset/services/gbif.py`) — **no** son coordenadas GPS reales de cámaras.
- **`media.timestamp`** se lee de la etiqueta EXIF `DateTimeOriginal`/`DateTime` de cada
  imagen cuando está presente. Las imágenes sin fecha EXIF legible (el dataset de
  ejemplo incluido no tiene ninguna) reciben una marca de tiempo interpolada dentro del
  rango EXIF real de ese mismo despliegue, o — si *ninguna* imagen de la partición tiene
  EXIF en absoluto — repartida a lo largo de un año de referencia fijo (2023). Cada fila
  estimada se señala en `media.mediaComments`
  (`"timestamp estimated: no EXIF datetime found in the source image"`); las filas con
  EXIF real dejan esa columna en blanco.
- **`deploymentStart`/`deploymentEnd`** son el mínimo/máximo de las marcas de tiempo
  *resueltas* de esa partición (EXIF real si existe, si no el año de referencia) —
  `deployments.deploymentComments` indica cuándo el rango de un despliegue incluye una
  estimación.

Todo lo demás se deriva directamente del dataset (especies a partir de los nombres de
clase YOLO, recuentos por imagen/por especie a partir de los cuadros de las etiquetas,
`classificationMethod=machine` porque las detecciones proceden de un modelo YOLO, no de
un revisor humano) o proviene de ajustes que ya controlas tú (licencia, contacto,
institución — sección 6).

## 4. Cómo lo generamos — cada fichero explicado

Todo lo siguiente se escribe en `--output-dir` (por defecto:
`<Documents>/donadataset/GBIF`).

### `deployments.csv`

Una fila por cada partición presente en los datos: `deploymentID`/`locationID` (el
nombre de la partición), `locationName`, `latitude`/`longitude`,
`deploymentStart`/`deploymentEnd`, `deploymentComments`.

### `media.csv`

Una fila por imagen: `mediaID` (el id de la imagen), `deploymentID`, `captureMethod`
(`activityDetection`), `timestamp`, `filePath`/`fileName`, `fileMediatype` (a partir de
la extensión del fichero), `mediaComments`.

Por defecto `filePath` es una ruta relativa dentro de la estructura del dataset fuente
— no una URL resoluble, ya que GBIF no tiene noción de "descarga esto de HuggingFace".
Pasa `--link-media-to-huggingface --hf-repo-id <repo>` para apuntarlo en su lugar al
**shard `.tar`** real (la salida de `huggingface prepare`,
`data/<split>/<split>-NNNNN.tar`) en el que se empaquetó esa imagen — `donadataset`
descarga solo `manifest.csv` del repo publicado (un fichero pequeño, no los propios
shards) para construir el mapeo `image_id → shard`. Esto **no** es una URL por foto:
todas las imágenes dentro del mismo shard comparten el mismo `filePath` (las imágenes de
train apuntan al shard de train, las de val al de val, y así sucesivamente — una
partición también puede repartirse entre varios ficheros `.tar` si es grande, en cuyo
caso las imágenes de esa partición apuntan al shard en el que realmente terminaron).
`mediaComments` siempre lo deja claro
(`"filePath points to the .tar shard containing <file>.jpg on HuggingFace Hub, not an
individually downloadable file"`) para que nadie lo confunda con un enlace directo a la
imagen. Esto requiere que `huggingface prepare`/`upload` ya hayan publicado
`manifest.csv` para el **mismo** dataset fuente — un desajuste (una imagen que `prepare`
ve localmente pero que no está en el manifest publicado) falla de forma ruidosa en vez
de adivinar en silencio.

### `observations.csv`

Una fila por imagen + especie con al menos un cuadro (`count` = número de cuadros de
esa especie en esa imagen — una foto con 3 cuadros de la misma especie es **una** fila
con `count=3`, no tres filas casi duplicadas). Las imágenes cuya única etiqueta es la
clase `Empty` del dataset fuente (o que no tienen ningún cuadro) reciben una única fila
`observationType=blank` en vez de descartarse en silencio. Columnas fijas:
`observationID`, `deploymentID`, `mediaID`, `eventID` (= `mediaID`, una foto es un
evento), `eventStart`/`eventEnd`, `observationLevel` (`media`), `observationType`
(`animal`/`blank`), `scientificName`, `count`, `classificationMethod` (`machine`),
`classifiedBy`.

### `datapackage.json`

El descriptor Frictionless: título/descripción/licencia/colaboradores (de los ajustes
`gbif`), `project` (diseño de muestreo, método de captura), cobertura
`spatial`/`temporal` derivada de los despliegues, `taxonomic` (cada especie distinta
observada), y un array `resources` que describe los tres CSVs (un esquema mínimo en
línea — solo nombres de campo, no el esquema oficial completo y restringido de la tabla
Camtrap DP).

### `<dataset-slug>-camtrap-dp.zip`

Los cuatro ficheros de arriba, comprimidos juntos — este es el único fichero que subes a
un IPT o alojas tú mismo para `gbif register`. Pasa `--upload-to-huggingface` (y
`--hf-repo-id`, por defecto `huggingface.repo_id`) para que `prepare` lo suba como un
fichero extra al repositorio dataset ya publicado en HuggingFace Hub justo después de
construirlo, para que obtengas una URL persistente
(`https://huggingface.co/datasets/<repo_id>/resolve/main/<slug>-camtrap-dp.zip`) sin
alojarlo en ningún otro sitio — consulta la sección 5b. Esto necesita un `HF_TOKEN` con
acceso de escritura y que el repo ya exista (`huggingface prepare` + `upload` ya
ejecutados); solo añade este único fichero, no toca nada más del repo.

## 5. Publicar — dos formas de meter el paquete en GBIF

### Configuración inicial

1. Crea una cuenta en [gbif.org](https://www.gbif.org) y solicita una cuenta de
   **organización** para WildINTEL (o usa el nodo GBIF existente de la Universidad de
   Huelva).
2. Instala el [GBIF IPT](https://www.gbif.org/ipt) v3+ (o usa una instancia alojada) si
   vas a publicar manualmente (5a abajo), o registra una **instalación** (de
   cualquier tipo — no tiene que ser un IPT) si vas a publicar mediante la vía de la
   API de Registry de `gbif register` (5b abajo).

### 5a. A través de un IPT (manual)

1. Ejecuta `donadataset publish gbif prepare`.
2. Abre tu **IPT v3+** (las versiones anteriores no soportan Camtrap DP), crea/actualiza
   un recurso, y sube `<dataset-slug>-camtrap-dp.zip` como su fuente.
3. Publica el recurso desde la interfaz del IPT. GBIF lo indexa en 24–48 horas y le
   asigna un DOI.

### 5b. A través de la API de Registry (con script, sin IPT)

El propio IPT no tiene API de subida (una
[petición de la comunidad para tenerla](https://github.com/gbif/ipt/issues/1249) se
cerró como `Won't-fix`), pero la API de Registry independiente de GBIF te permite
registrar un dataset y apuntarlo a un archivo que alojas tú mismo. El alojamiento más
sencillo es el repositorio de HuggingFace Hub que ya has publicado:

```bash
export HF_TOKEN=your-hf-write-token
donadataset publish gbif prepare --upload-to-huggingface --link-media-to-huggingface
# ↑ imprime la URL persistente: https://huggingface.co/datasets/<repo_id>/resolve/main/donadataset-camtrap-dp.zip

export GBIF_USERNAME=your-gbif-org-username
export GBIF_PASSWORD=your-gbif-org-password
donadataset publish gbif register --archive-url https://huggingface.co/datasets/<repo_id>/resolve/main/donadataset-camtrap-dp.zip
```

`--link-media-to-huggingface` (independiente de `--upload-to-huggingface` — puedes usar
cualquiera de los dos por separado) hace que `media.filePath` dentro del paquete apunte
a los shards `.tar` reales ya presentes en HuggingFace Hub en vez de a una ruta relativa
local; consulta la sección 4.

(O aloja `<dataset-slug>-camtrap-dp.zip` en cualquier otro sitio que prefieras y omite
`--upload-to-huggingface` — a `register` solo le importa que `--archive-url` sea una
URL pública y accesible por GBIF.)

La primera ejecución crea el dataset y añade un endpoint `CAMTRAP_DP` apuntando a
`--archive-url`; registra el UUID del dataset devuelto en
`gbif_linked_dataset_record.json` dentro de `--output-dir`, y cada ejecución posterior
lee ese fichero, actualiza los metadatos del dataset, y reemplaza el endpoint. Usa
`--environment sandbox` (por defecto) para probar antes de `--environment production`, y
`--dry-run` para previsualizar sin llamar a la API.

**Requisitos previos únicos para 5b:** una cuenta en GBIF.org
(`GBIF_USERNAME`/`GBIF_PASSWORD` — Basic Auth, no un token), y una **organización** +
**instalación** ya registradas en el Registry de GBIF (la instalación no tiene que ser
un IPT). Configura sus UUIDs una vez con `gbif config set
publishing_organization_key=...` / `installation_key=...`. Las credenciales tampoco
hace falta exportarlas en cada sesión — `GBIF_USERNAME`/`GBIF_PASSWORD` siempre ganan si
están definidas, pero si no, recurren a `gbif.username`/`gbif.password` en
`settings.toml`, guardados con `gbif config set username` / `config set password`
(entrada oculta, nunca mostrada ni por `config show`).

## 6. Configuración

```bash
donadataset publish gbif config show
donadataset publish gbif config set contact_email=you@example.org
donadataset publish gbif config wizard
```

`rights_holder`, `institution_code`, `contact_name`, `contact_email` (sin definir por
defecto), y `classified_by` alimentan los colaboradores/observaciones de
`datapackage.json`; los campos de licencia alimentan su array `licenses`.
`environment`, `publishing_organization_key`, `installation_key`, y
`registry_language` solo los usa `register` (sección 5b) — `prepare` los ignora.

## 7. En cada nueva versión

Vuelve a ejecutar `prepare` — siempre regenera todo el paquete a partir del dataset
actual, así que no hay nada que mantener sincronizado a mano. Luego, o bien vuelve a
subir el nuevo `.zip` al mismo recurso del IPT y provoca un nuevo rastreo (5a), o vuelve
a ejecutar `register` con la misma (re-subida) `--archive-url` (5b) — GBIF vuelve a
rastrear un endpoint modificado automáticamente en unas pocas horas.
