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
[HuggingFace Hub](publishing-huggingface.md). `media.filePath` siempre apunta ahí: la
URL persistente del shard `.tar` real (la salida de `huggingface prepare`) en el que se
empaquetó cada imagen — consulta la entrada de `media.csv` de la sección 5 para ver
exactamente a qué apunta esa URL (no es una URL por foto).

## 3. De dónde sale el dataset fuente

Generar el paquete Camtrap DP implica leer la fecha EXIF de cada imagen y contar las
cajas de cada label — a diferencia de los pequeños ficheros de evidencia que Zenodo
re-publica, esos datos solo existen dentro del dataset completo, así que `prepare`
necesita las imágenes/labels reales en disco — pero también necesita el propio
`manifest.csv`, ya que `media.filePath` siempre enlaza a HuggingFace Hub (ver sección 5)
y ese es el único registro de qué shard `.tar` empaquetó cada imagen.

Por defecto (sin `--source-dataset-dir`), los obtiene de HuggingFace Hub
(`--hf-repo-id`, el mismo valor por defecto que `repo_id` en `huggingface prepare`):

1. Reutiliza `--output-dir/hfh_download/` tal cual si una ejecución anterior ya
   extrajo ahí el dataset — sin descarga, sin llamada de red, sin volver a extraer.
2. Si no, reutiliza el export local en `<Documents>/donadataset/HFH/<repo_id>` si
   `huggingface prepare` ya dejó uno ahí — tampoco hace falta descarga.
3. Si no, descarga el repo publicado (incluidos los shards `.tar`) directamente a
   `--output-dir/hfh_download/` (por defecto
   `<Documents>/donadataset/GBIF/<repo_id>/hfh_download/`).
4. En cualquier caso, los shards `.tar` se extraen dentro de ese mismo directorio
   `hfh_download/` (in situ, si es ahí donde se acaban de descargar).

`hfh_download/` vive *dentro* de `--output-dir` — `--overwrite` solo borra los
ficheros del paquete Camtrap DP en sí (los tres CSV, `datapackage.json`, el `.zip`),
nunca este directorio, así que sobrevive entre ejecuciones sin importar `--overwrite`.

Pasa `--source-dataset-dir` para saltarte esta resolución y apuntar directamente a una
carpeta YOLO local — igualmente tiene que ser (o contener una copia de) un export real
de `huggingface prepare`, con `manifest.csv` incluido, ya que `--hf-repo-id` sigue
siendo obligatorio y `media.filePath` sigue enlazando siempre a HuggingFace Hub; un
output de `generate real` sin pasar nunca por `huggingface prepare` no funciona aquí.

## 4. Qué inventa `gbif prepare`, y por qué

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
clase YOLO, recuentos por imagen/por especie a partir de los cuadros de las etiquetas) o
proviene de ajustes que ya controlas tú (licencia, contacto, institución — sección 7).
`classificationMethod` siempre vale `human` y `classifiedBy` siempre vale `WildINTEL
experts` — fijos, no configurables, porque cada etiqueta de este dataset procede de
revisión humana, no de un modelo automático.

## 5. Cómo lo generamos — cada fichero explicado

Todo lo siguiente se escribe en `--output-dir` (por defecto:
`<Documents>/donadataset/GBIF/<repo_id>`, o `<Documents>/donadataset/GBIF` a secas si
`huggingface.repo_id` todavía no está configurado).

### `deployments.csv`

Una fila por cada partición presente en los datos: `deploymentID`/`locationID` (el
nombre de la partición), `locationName`, `latitude`/`longitude`,
`deploymentStart`/`deploymentEnd`, `deploymentComments`.

### `media.csv`

Una fila por imagen: `mediaID` (el id de la imagen), `deploymentID`, `captureMethod`
(`activityDetection`), `timestamp`, `filePath`/`fileName`, `fileMediatype` (a partir de
la extensión del fichero), `mediaComments`.

`filePath` siempre apunta al **shard `.tar`** real (la salida de `huggingface prepare`,
`data/<split>/<split>-NNNNN.tar`) en el que se empaquetó esa imagen — nunca a una ruta
relativa local, ya que GBIF no tiene noción de "descarga esto de HuggingFace" y no hay
ningún flag para desactivarlo. `donadataset` lee `manifest.csv` de donde sea que se haya
resuelto el dataset fuente (sección 3) — en local, sin llamada de red — para construir
el mapeo `image_id → shard`. Esto **no** es una URL por foto: todas las imágenes dentro
del mismo shard comparten el mismo `filePath` (las imágenes de train apuntan al shard de
train, las de val al de val, y así sucesivamente — una partición también puede
repartirse entre varios ficheros `.tar` si es grande, en cuyo caso las imágenes de esa
partición apuntan al shard en el que realmente terminaron). `mediaComments` siempre lo
deja claro (`"filePath points to the .tar shard containing <file>.jpg on HuggingFace
Hub, not an individually downloadable file"`) para que nadie lo confunda con un enlace
directo a la imagen. Esto requiere que el dataset fuente tenga un `manifest.csv` para
las **mismas** imágenes que se están escaneando — un desajuste (una imagen que
`prepare` ve localmente pero que no está en el manifest) falla de forma ruidosa en vez
de adivinar en silencio.

`fileName` es la ruta *dentro* de ese shard (p.ej. `images/train/img_001.jpg`,
coincidiendo con el `arcname` que le dio `huggingface prepare` al empaquetar el `.tar`)
— no solo el nombre suelto del fichero — ya que eso es lo que hace falta de verdad para
localizar la imagen una vez alguien descargue y extraiga el shard al que apunta
`filePath`.

### `observations.csv`

Una fila por imagen + especie con al menos un cuadro (`count` = número de cuadros de
esa especie en esa imagen — una foto con 3 cuadros de la misma especie es **una** fila
con `count=3`, no tres filas casi duplicadas). Las imágenes cuya única etiqueta es la
clase `Empty` del dataset fuente (o que no tienen ningún cuadro) reciben una única fila
`observationType=blank` en vez de descartarse en silencio. Columnas fijas:
`observationID`, `deploymentID`, `mediaID`, `eventID` (= `mediaID`, una foto es un
evento), `eventStart`/`eventEnd`, `observationLevel` (`media`), `observationType`
(`animal`/`blank`), `scientificName`, `count`, `classificationMethod` (siempre `human`),
`classifiedBy` (siempre `WildINTEL experts`).

### `datapackage.json`

El descriptor Frictionless: título/descripción/licencia/colaboradores (de los ajustes
`gbif`), `project` (diseño de muestreo, método de captura), cobertura
`spatial`/`temporal` derivada de los despliegues, `taxonomic` (cada especie distinta
observada), y un array `resources` que describe los tres CSVs (un esquema mínimo en
línea — solo nombres de campo, no el esquema oficial completo y restringido de la tabla
Camtrap DP).

### `<dataset-slug>-camtrap-dp.zip`

`<dataset-slug>` no es un flag propio — es el segmento de dataset de `--hf-repo-id`
(`user_or_org/dataset` → `dataset`), la identidad real de lo que se está empaquetando.
Solo recurre a `HUGGINGFACE.dataset_slug` si no hay repo_id disponible en absoluto (p.ej.
`--source-dataset-dir` usado sin ningún repo_id configurado).

Los cuatro ficheros de arriba, comprimidos juntos — este es el único fichero que subes a
un IPT o alojas tú mismo para `gbif register`. Ejecuta `donadataset publish gbif upload`
después (consulta la sección 6b) para subirlo como un fichero extra al repositorio
dataset ya publicado en HuggingFace Hub, para que obtengas una URL persistente
(`https://huggingface.co/datasets/<repo_id>/resolve/main/<slug>-camtrap-dp.zip`) sin
alojarlo en ningún otro sitio. `upload` copia el `.zip` a un export local de HuggingFace
Hub y regenera el `checksums-sha256.txt` de ese export antes de subir — ambos ficheros,
acotados, así no se vuelve a subir nada más de lo ya publicado.

`upload` resuelve ese export local de la misma forma "primero local, si no descarga y
cachea" con la que `prepare` resuelve su dataset fuente (sección 3): reutiliza
`<Documents>/donadataset/HFH/<repo_id>` (la salida de `huggingface prepare`/`upload`) si
ya está ahí, y si no, descarga el repo ya publicado en
`<Documents>/donadataset/GBIF/<repo_id>/hfh_download` (el mismo directorio caché que ya
usa `prepare`) y reutiliza esa caché en ejecuciones posteriores. Pasa `--hfh-output-dir`
para saltarte esta resolución y apuntar a un directorio concreto. `--dry-run` nunca
descarga — si ninguna de las dos ubicaciones tiene el export todavía, solo informa de que
lo haría. Esto necesita un `HF_TOKEN` con acceso de escritura y que el repo ya exista
(`huggingface prepare` + `upload` ya ejecutados).

## 6. Publicar — dos formas de meter el paquete en GBIF

### Configuración inicial

1. Crea una cuenta en [gbif.org](https://www.gbif.org) y solicita una cuenta de
   **organización** para WildINTEL (o usa el nodo GBIF existente de la Universidad de
   Huelva).
2. Instala el [GBIF IPT](https://www.gbif.org/ipt) v3+ (o usa una instancia alojada) si
   vas a publicar manualmente (6a abajo), o registra una **instalación** (de
   cualquier tipo — no tiene que ser un IPT) si vas a publicar mediante la vía de la
   API de Registry de `gbif register` (6b abajo).

### 6a. A través de un IPT (manual)

1. Ejecuta `donadataset publish gbif prepare`.
2. Abre tu **IPT v3+** (las versiones anteriores no soportan Camtrap DP), crea/actualiza
   un recurso, y sube `<dataset-slug>-camtrap-dp.zip` como su fuente.
3. Publica el recurso desde la interfaz del IPT. GBIF lo indexa en 24–48 horas y le
   asigna un DOI.

### 6b. A través de la API de Registry (con script, sin IPT)

El propio IPT no tiene API de subida (una
[petición de la comunidad para tenerla](https://github.com/gbif/ipt/issues/1249) se
cerró como `Won't-fix`), pero la API de Registry independiente de GBIF te permite
registrar un dataset y apuntarlo a un archivo que alojas tú mismo. El alojamiento más
sencillo es el repositorio de HuggingFace Hub que ya has publicado:

```bash
donadataset publish gbif prepare

export HF_TOKEN=your-hf-write-token
donadataset publish gbif upload
# ↑ imprime la URL persistente: https://huggingface.co/datasets/<repo_id>/resolve/main/donadataset-camtrap-dp.zip

export GBIF_USERNAME=your-gbif-org-username
export GBIF_PASSWORD=your-gbif-org-password
donadataset publish gbif register --archive-url https://huggingface.co/datasets/<repo_id>/resolve/main/donadataset-camtrap-dp.zip
```

(O aloja `<dataset-slug>-camtrap-dp.zip` en cualquier otro sitio que prefieras y omite
`gbif upload` — a `register` solo le importa que `--archive-url` sea una URL pública y
accesible por GBIF.)

`donadataset publish gbif pipeline` encadena los tres (`prepare` -> `upload` ->
`register`) de un tirón.

La primera ejecución crea el dataset y añade un endpoint `CAMTRAP_DP` apuntando a
`--archive-url`; registra el UUID del dataset devuelto en
`gbif_linked_dataset_record.json` dentro de `--output-dir`, y cada ejecución posterior
lee ese fichero, actualiza los metadatos del dataset, y reemplaza el endpoint. Usa
`--environment sandbox` (por defecto) para probar antes de `--environment production`, y
`--dry-run` para previsualizar sin llamar a la API.

**Requisitos previos únicos para 6b:** una cuenta en GBIF.org
(`GBIF_USERNAME`/`GBIF_PASSWORD` — Basic Auth, no un token), y una **organización** +
**instalación** ya registradas en el Registry de GBIF (la instalación no tiene que ser
un IPT). Configura sus UUIDs una vez con `gbif config set
publishing_organization_key=...` / `installation_key=...`. Las credenciales tampoco
hace falta exportarlas en cada sesión — `GBIF_USERNAME`/`GBIF_PASSWORD` siempre ganan si
están definidas, pero si no, recurren a `gbif.username`/`gbif.password` en
`settings.toml`, guardados con `gbif config set username` / `config set password`
(entrada oculta, nunca mostrada ni por `config show`).

## 7. Configuración

```bash
donadataset publish gbif config show
donadataset publish gbif config set contact_email=you@example.org
donadataset publish gbif config wizard
```

`institution_code` y `contact_email` (sin definir por defecto) alimentan los
colaboradores de `datapackage.json`. `environment`, `publishing_organization_key`,
`installation_key`, y `registry_language` solo los usa `register` (sección 6b) —
`prepare` los ignora.

Deliberadamente **no** están aquí: `dataset_name`, `description`, los campos de
licencia, el nombre de la organización en los colaboradores, ni el nombre de contacto
a mostrar. Esos salen directamente de `HUGGINGFACE.dataset_name`/`description`/
`license_id`/`license_name`/`license_url`/`author_affiliation`/`author_family_names`
— la misma identidad que ya usa `huggingface prepare` (`donadataset publish
huggingface config set ...`) — así el paquete de GBIF no puede desincronizarse en
silencio de lo que de verdad está publicado en HuggingFace Hub. `classificationMethod`/
`classifiedBy` tampoco son configurables — cada etiqueta de este dataset procede de
revisión humana, así que `prepare` siempre escribe `human`/`WildINTEL experts`.

## 8. En cada nueva versión

Vuelve a ejecutar `prepare` — siempre regenera todo el paquete a partir del dataset
actual, así que no hay nada que mantener sincronizado a mano. Luego, o bien vuelve a
subir el nuevo `.zip` al mismo recurso del IPT y provoca un nuevo rastreo (6a), o vuelve
a ejecutar `register` con la misma (re-subida) `--archive-url` (6b) — GBIF vuelve a
rastrear un endpoint modificado automáticamente en unas pocas horas.
