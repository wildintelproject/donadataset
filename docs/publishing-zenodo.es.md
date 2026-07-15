# Publicar en Zenodo

Esta guía explica cómo se publica DonaDataset en **Zenodo** usando la CLI de
`donadataset`. Está dirigida al **mantenedor del dataset**.

---

## 1. Qué es Zenodo

[Zenodo](https://zenodo.org) es un repositorio de datos de investigación de acceso
abierto operado por el CERN (Ginebra, Suiza). Los investigadores lo usan para archivar
y compartir cualquier resultado de investigación — datasets, software, artículos,
presentaciones — y asigna un **DOI** (Digital Object Identifier) permanente a cada
registro publicado. Zenodo también ofrece una instancia **Sandbox**
(`sandbox.zenodo.org`) con una API idéntica, usada para pruebas sin tocar DOIs reales.

A diferencia de HuggingFace Hub, Zenodo no está diseñado como un servicio de
alojamiento masivo de ficheros para datasets de machine learning — es un **archivo
citable a largo plazo**. Ese es exactamente el papel que juega aquí: no compite con
HuggingFace Hub, lo complementa.

## 2. Qué permite subir Zenodo

Un registro de Zenodo puede técnicamente contener ficheros de cualquier tipo, incluidos
datos binarios grandes. **En este proyecto deliberadamente no lo usamos así.**
DonaDataset sigue el propio patrón recomendado por Zenodo para datasets ya alojados en
otro sitio: un **registro de dataset enlazado** — Zenodo aloja solo los metadatos y la
evidencia que prueban que el dataset existe, está verificado, y dónde encontrarlo,
mientras que las imágenes/etiquetas reales permanecen en HuggingFace Hub (ver la
[guía de HuggingFace](publishing-huggingface.md)).

Esto se refuerza mediante `related_identifiers` en los metadatos del registro de
Zenodo: enlaces explícitos a la página del dataset en HuggingFace y a su árbol de
ficheros, para que cualquiera que llegue a la página del DOI vea inmediatamente "los
datos viven aquí".

## 3. Qué subimos

`donadataset publish zenodo prepare` **no** necesita una copia local de la exportación
de HuggingFace para obtener los ficheros de evidencia. En su lugar:

1. Se conecta a HuggingFace Hub (`--repo-id`) y **descarga el repositorio en vivo**, en
   su propio directorio (`--output-dir`).
2. Verifica esa descarga contra los mismos checksums/manifest con los que se construyó
   la exportación (checksums globales + hashes internos de los miembros `.tar`).
3. Extrae los pequeños ficheros de evidencia de esa misma copia descargada (nunca los
   shards `.tar` — esos permanecen exclusivos de HuggingFace) y los sube a un nuevo
   depósito de Zenodo.
4. Reserva un DOI para ese depósito.

Así que "qué subimos" es: **los ficheros de evidencia ya presentes en el repositorio de
HuggingFace** (los mismos descritos en la
[guía de HuggingFace](publishing-huggingface.md#4-como-lo-subimos-cada-fichero-explicado)),
más un nuevo report generado por este paso de descarga-y-verificación, más un registro
JSON que ata todo junto. Nada de esto se escribe a mano — todo lo producen los comandos
de la sección 5.

## 4. Cómo lo subimos — cada fichero explicado

### Ficheros de evidencia obtenidos de la descarga en vivo de HuggingFace

Son los mismos ficheros ya descritos en la
[guía de HuggingFace](publishing-huggingface.md#4-como-lo-subimos-cada-fichero-explicado) —
`README.md`, `LICENSE`, `CITATION.cff`, `HuggingFaceHub.yaml`, `donana.yaml`,
`dataset_info.json`, `metadata.csv`, `manifest.csv`, `manifest-files-sha256.csv`,
`checksums-sha256.txt`, `validation_report.json`, `verification_report_local.json` —
simplemente obtenidos frescos de HuggingFace Hub en vez de leídos de una carpeta local
previa a la subida. Esto garantiza que lo que recibe Zenodo coincide exactamente con lo
que hay público en este momento, no con lo que tu máquina tuviera en disco en algún
momento anterior.

### `verification_report_downloaded.json`

El resultado del paso de descarga-y-verificación en vivo descrito arriba:
`status: "passed"/"failed"`, `repo_id`, la ruta local donde se escribió la descarga, y
los recuentos/errores de volver a comprobar los checksums globales y los hashes
internos de los `.tar`. Se genera **fresco cada vez** que se ejecuta `zenodo prepare`
(nunca se reutiliza de una ejecución anterior de `huggingface download`), y se sube a
su vez a Zenodo como una pieza más de evidencia.

### `zenodo_linked_dataset_record.json`

El **resultado** de crear el depósito de Zenodo — se escribe localmente tras el éxito
de `prepare`, y también se sube a Zenodo como un fichero en el mismo depósito:
`deposition_id`, `record_id`, `reserved_doi`, `doi_url`, `record_url`, el entorno de
Zenodo (`sandbox`/`production`), y un bloque `huggingface_verification` que resume el
paso de descarga-y-verificación de arriba. Los comandos posteriores (`upload`,
`check-readiness`, `release`) leen todos este fichero para saber con qué depósito
existente están trabajando.

### `zenodo_deposition_response.json` / `zenodo_file_verification_report.json` / `zenodo_link_verification_report.json`

Rastro de auditoría solo local (no se sube a Zenodo): la respuesta cruda de la API del
depósito tras subir los ficheros, una comprobación de que el tamaño/hash de cada
fichero subido coincide con lo que Zenodo reporta, y una comprobación de que las URLs
de `related_identifiers` (los enlaces a HuggingFace) resuelven de verdad.

### `metadata_update_report.json` / `zenodo_publish_response.json` / `zenodo_publication_report.json` / `public_release_readiness_report.json`

Escritos por los pasos posteriores de la sección 5 (`upload`, `release`,
`check-readiness` respectivamente) — no forman parte de la subida inicial de
`prepare`, pero viven en el mismo directorio para tener un único rastro de auditoría
completo de toda la publicación.

## 5. Comandos para publicar

### Configuración inicial

1. Crea una cuenta en [zenodo.org](https://zenodo.org) (usa primero
   `sandbox.zenodo.org` para probar todo el flujo sin generar un DOI real).
2. Crea un token de acceso personal (**Applications → Personal access tokens**, con los
   permisos `deposit:write` + `deposit:actions`).
3. Configúralo como `ZENODO_TOKEN`, o guárdalo una vez mediante `donadataset publish
   zenodo config set token`.

### Publicar una nueva versión

La publicación en Zenodo ocurre **después** de que el dataset ya esté en HuggingFace
Hub (pasos 1–4 de la
[guía de HuggingFace](publishing-huggingface.md#5-comandos-para-publicar)), y el paso
final e irreversible de "publicar" ocurre **después** de que HuggingFace Hub se haya
hecho público:

```bash
# 1. Crear el draft de Zenodo, descargar+verificar HuggingFace Hub en vivo, subir la
#    evidencia, y reservar un DOI. Usa sandbox.zenodo.org por defecto (ver templates/Zenodo.yaml.j2).
export ZENODO_TOKEN='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
donadataset publish zenodo prepare --repo-id <tu-usuario>/<dataset-slug>

# 2. Insertar el DOI reservado en los metadatos de tu exportación local de HuggingFace
#    (CITATION.cff, dataset_info.json, README.md) y recalcular los checksums
donadataset publish zenodo upload --hfh-output-dir <export-dir>

# 3. Volver a subir a HuggingFace Hub para que el DOI se refleje en los ficheros públicos
donadataset publish huggingface upload

# 4. Hacer público el repositorio de HuggingFace (si no se ha hecho ya)
donadataset publish huggingface release

# 5. Comprobación final de seguridad de solo lectura antes de publicar de verdad
donadataset publish zenodo check-readiness --repo-id <tu-usuario>/<dataset-slug>

# 6. Publicar el draft de Zenodo — IRREVERSIBLE, hazlo solo cuando el paso 5 pase
donadataset publish zenodo release --repo-id <tu-usuario>/<dataset-slug>
```

O ejecuta los pasos 1, 2, 5 y 6 de una vez con `pipeline` (se pausa entre `upload` y
`check-readiness` para que puedas hacer los pasos 3–4 en medio):

```bash
donadataset publish zenodo pipeline --repo-id <tu-usuario>/<dataset-slug>
```

### La forma fácil: `wizard`

```bash
donadataset publish zenodo wizard
```

Te guía por los mismos seis pasos de forma interactiva, uno cada vez. A diferencia de
`pipeline` (que falla en seco ante el primer error y siempre crea un draft nuevo),
`wizard`:

- Pide `--repo-id` si todavía no has configurado uno (mismo prompt que
  `donadataset publish huggingface wizard`, comparte el mismo valor de
  `settings.toml`), y ofrece guardarlo para no volver a preguntarlo.
- Detecta un draft de Zenodo enlazado ya existente en `--output-dir` y pregunta si
  sincronizarlo (`--sync-existing-draft`) en vez de crear siempre uno nuevo.
- Hace el paso 3 (re-subida a HuggingFace Hub) por ti automáticamente, en vez de
  dejarlo como un paso manual que tienes que recordar.
- Te avisa antes de tocar `production` si `zenodo.environment` no es `sandbox`.
- Pide confirmación explícita antes de la publicación final — el único paso
  irreversible.
- Si algún paso falla (fallo de red, error transitorio de la API...), te deja
  reintentarlo o abortar en vez de simplemente salir.

No confundir con `donadataset publish zenodo config wizard`, que solo edita campos de
`settings.toml` y no publica nada.

Ninguno de estos comandos tiene flag `--config`: siempre renderizan la única plantilla
Jinja2 incluida (`templates/Zenodo.yaml.j2`) usando `--repo-id`/`--output-dir` (y
`--hfh-output-dir` para `upload`), cuyos valores por defecto vienen de `settings.toml`
— configúralos una vez con `donadataset publish zenodo config set <campo>=...` (o
`config wizard`) y evita repetir `--environment`/etc. en cada ejecución. Todos los
comandos aceptan `--dry-run` (excepto `download`, que no existe como paso de
publicación — usa `donadataset publish zenodo download` para *recuperar* un registro ya
publicado, y `pipeline`, que siempre se ejecuta de verdad). El *nombre de la variable
de entorno* del token está fijado a `ZENODO_TOKEN` en la plantilla (no es un setting ni
un flag) — pero su *valor* no hace falta exportarlo en cada sesión: `export
ZENODO_TOKEN=...` siempre gana si está definida, si no recurre a `zenodo.token` en
`settings.toml`, guardado con `donadataset publish zenodo config set token` (entrada
oculta, nunca mostrada ni por `config show`).

> ⚠️ **`zenodo release` es el único paso que publica de verdad.** Todo lo anterior
> (`prepare`, `upload`, `check-readiness`) solo crea/actualiza un **draft** — puedes
> volver a ejecutarlos tantas veces como necesites. Una vez que `release` tiene éxito,
> el registro de Zenodo y sus ficheros ya no se pueden editar ni eliminar.
