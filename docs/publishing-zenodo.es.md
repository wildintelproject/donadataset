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

Tres comandos separados dividen "preparar en local" de "hablar de verdad con
Zenodo" — el mismo convenio que ya usan `huggingface prepare`/`upload`:

- **`prepare`** nunca sube nada a Zenodo. Tampoco necesita una copia local de la
  exportación de HuggingFace — en su lugar:
  1. Se conecta a HuggingFace Hub (`--repo-id`) y **descarga el repositorio en vivo**,
     en su propio directorio (`--output-dir`).
  2. Verifica esa descarga contra los mismos checksums/manifest con los que se
     construyó la exportación (checksums globales + hashes internos de los `.tar`).
  3. Crea (o, con `--sync-existing-draft`, relee) un depósito de Zenodo solo para
     **reservar/leer un DOI** — todavía no se sube ningún fichero.
  4. Copia los pequeños ficheros de evidencia de esa copia descargada (nunca los
     shards `.tar` — esos permanecen exclusivos de HuggingFace) a `--output-dir`,
     inyectando el DOI reservado en el `CITATION.cff` de esa copia y recalculando el
     `checksums-sha256.txt` de ese directorio para que siga siendo consistente.
- **`upload`** lee exactamente lo que `prepare` dejó en `--output-dir` y lo sube, tal
  cual, al bucket del depósito de Zenodo — es el único comando que habla de verdad con
  la API de ficheros de Zenodo.
- **`sync-doi`** refleja ese DOI en el lado de HuggingFace Hub: copia el `CITATION.cff`
  con el DOI ya inyectado sobre la copia propia de la exportación de HuggingFace,
  actualiza la sección "## Zenodo DOI" del `README.md` de esa exportación, recalcula su
  `checksums-sha256.txt`, y (por defecto) vuelve a subir solo esos tres ficheros a
  HuggingFace Hub — ver el recuadro de `--verify-data` en la sección 4 y la secuencia
  completa de comandos en la sección 5.

Así que "qué subimos a Zenodo" es: **los ficheros de evidencia ya presentes en el
repositorio de HuggingFace** (los mismos descritos en la
[guía de HuggingFace](publishing-huggingface.md#4-como-lo-subimos-cada-fichero-explicado)),
con `CITATION.cff` llevando el propio DOI de Zenodo, más un nuevo report generado por
el paso de descarga-y-verificación, más un registro JSON que ata todo junto. Nada de
esto se escribe a mano — todo lo producen los comandos de la sección 5.

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

`prepare` copia todo esto a `--output-dir` tal cual, **excepto** `CITATION.cff` y
`README.md`: esas dos copias reciben el DOI que Zenodo acaba de reservar (los campos
`doi`/`url` de `CITATION.cff`, o una entrada en `identifiers` en Sandbox; `README.md`
recibe una sección "## Zenodo DOI", igual que recibe la copia de HuggingFace vía
`sync-doi` — ver más abajo), y `checksums-sha256.txt` se regenera para `--output-dir`
justo después para que siga coincidiendo con esos ficheros ya modificados que tiene al
lado — es decir, la copia de `checksums-sha256.txt` que aloja Zenodo difiere
intencionadamente de la de HuggingFace (describe el conjunto plano de evidencia de
`--output-dir`, no la exportación completa de HuggingFace).

### `<dataset-slug>-camtrap-dp.zip` (opcional)

Si has publicado en [GBIF](publishing-gbif.md) con `gbif prepare` + `gbif
upload`, el paquete Camtrap DP resultante queda en la raíz del repo de
HuggingFace Hub junto a los ficheros de arriba. `prepare` lo recoge automáticamente (por
patrón de nombre, `*-camtrap-dp.zip`, ya que se llama según el `--dataset-slug` propio de
GBIF) y lo incluye como un fichero de evidencia más — es metadata pequeña y estructurada
sin ninguna imagen dentro, así que encaja con el mismo criterio que ya sigue Zenodo para
`manifest.csv`. No falla nada si no está ahí; publicar en GBIF es completamente opcional
y esto se recoge de forma oportunista.

### `verification_report_downloaded.json`

El resultado del paso de descarga-y-verificación en vivo descrito arriba:
`status: "passed"/"failed"`, `repo_id`, la ruta local donde se escribió la descarga,
`data_verified` (si los shards `data/<split>/*.tar` también se descargaron y
verificaron — ver `--verify-data` abajo), y los recuentos/errores de volver a
comprobar los checksums globales y los hashes internos de los `.tar`. Se genera
**fresco cada vez** que se ejecuta `zenodo prepare` (nunca se reutiliza de una
ejecución anterior de `huggingface download`), y se sube a su vez a Zenodo como una
pieza más de evidencia.

> 💡 **`--verify-data` / `--no-verify-data` (por defecto: desactivado).** Zenodo nunca
> sube los shards `.tar` — solo los ficheros pequeños de evidencia — así que por
> defecto `prepare` ni siquiera los descarga de HuggingFace Hub para comprobarlos,
> solo los metadatos. Pasa `--verify-data` para además descargar cada shard y volver a
> calcular el hash de su contenido contra `manifest-files-sha256.csv`, como garantía
> extra de que las propias imágenes/etiquetas publicadas (no solo los metadatos que
> las describen) siguen coincidiendo con lo que escribió originalmente `huggingface
> prepare` — al coste de una descarga completa del dataset cada vez que ejecutes este
> comando.

### `zenodo_linked_dataset_record.json`

El **resultado** de crear el depósito de Zenodo — se escribe localmente en `prepare`
(`deposition_id`, `record_id`, `reserved_doi`, `doi_url`, `record_url`, el entorno de
Zenodo, y un bloque `huggingface_verification` que resume el paso de
descarga-y-verificación de arriba), luego se vuelve a escribir en `upload` una vez que
los ficheros se suben y verifican de verdad, y también se sube a Zenodo como un fichero
en el mismo depósito. Los comandos posteriores (`upload`, `check-readiness`, `release`)
leen todos este fichero para saber con qué depósito existente están trabajando.

### `zenodo_deposition_response.json` / `zenodo_file_verification_report.json` / `zenodo_link_verification_report.json`

Los escribe `upload` (el paso que habla de verdad con la API de ficheros de Zenodo)
como rastro de auditoría solo local (no se sube a Zenodo): la respuesta cruda de la API
del depósito tras subir los ficheros, una comprobación de que el tamaño/hash de cada
fichero subido coincide con lo que Zenodo reporta, y una comprobación de que las URLs
de `related_identifiers` (los enlaces a HuggingFace) resuelven de verdad.

### `zenodo_publish_response.json` / `zenodo_publication_report.json` / `public_release_readiness_report.json`

Escritos por los pasos posteriores de la sección 5 (`release`/`check-readiness`) — no
forman parte de `prepare`/`upload`, pero viven en el mismo directorio para tener un
único rastro de auditoría completo de toda la publicación.

## 5. Comandos para publicar

### Configuración inicial

1. Crea una cuenta en [zenodo.org](https://zenodo.org) (usa primero
   `sandbox.zenodo.org` para probar todo el flujo sin generar un DOI real).
2. Crea un token de acceso personal (**Applications → Personal access tokens**, con los
   permisos `deposit:write` + `deposit:actions`).
3. Configúralo como `ZENODO_TOKEN`, o guárdalo una vez mediante `donadataset publish
   zenodo config set token`.

### Publicar en comunidades de Zenodo (opcional)

`prepare` envía el depósito a una o varias [comunidades de
Zenodo](https://zenodo.org/communities) (colecciones curadas — enviarlo a una que no
curas tú deja el registro pendiente de aprobación; publicar el depósito en sí no se ve
afectado). Por defecto es solo `wildintelproject`; cambia la lista completa (separada
por comas) a lo que te convenga:

```bash
donadataset publish zenodo config set communities=camera-traps,biodiversity
```

(pasa un valor vacío, `communities=`, para no enviarlo a ninguna comunidad)

Cada ejecución de `prepare` las envía todas por defecto. Para limitar una ejecución
concreta a un subconjunto (nunca a algo fuera de esta lista — añádelo primero a
`ZENODO.communities` si falta), usa `--communities`:

```bash
donadataset publish zenodo prepare --repo-id <tu-usuario>/<dataset-slug> --communities camera-traps
```

Esto solo tiene efecto cuando `prepare` crea un depósito **nuevo** — `--sync-existing-draft`
relee un draft ya creado sin volver a enviar la metadata, así que las comunidades
fijadas al crearlo no cambian en una sincronización posterior.

### Publicar una nueva versión

La publicación en Zenodo ocurre **después** de que el dataset ya esté en HuggingFace
Hub (pasos 1–4 de la
[guía de HuggingFace](publishing-huggingface.md#5-comandos-para-publicar)), y el paso
final e irreversible de "publicar" ocurre **después** de que HuggingFace Hub se haya
hecho público:

```bash
# 1. Crear el draft de Zenodo, descargar+verificar HuggingFace Hub en vivo, reservar un
#    DOI, y preparar los ficheros de evidencia (con el DOI ya inyectado en
#    CITATION.cff) en --output-dir. Todavía no se sube nada a Zenodo. Usa
#    sandbox.zenodo.org por defecto (ver templates/Zenodo.yaml.j2).
export ZENODO_TOKEN='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
donadataset publish zenodo prepare --repo-id <tu-usuario>/<dataset-slug>

# 2. Subir al bucket del depósito de Zenodo exactamente lo que preparó 'prepare'.
donadataset publish zenodo upload --repo-id <tu-usuario>/<dataset-slug>

# 3. Reflejar el DOI en el lado de HuggingFace Hub: copia el CITATION.cff con el DOI
#    ya inyectado sobre la copia propia de la exportación de HuggingFace, actualiza la
#    sección "## Zenodo DOI" de su README.md, recalcula su checksums-sha256.txt, y
#    vuelve a subir SOLO esos tres ficheros a HuggingFace Hub automáticamente
#    (--no-upload para saltarte esa última parte).
donadataset publish zenodo sync-doi --hfh-output-dir <export-dir>

# 4. Hacer público el repositorio de HuggingFace (si no se ha hecho ya)
donadataset publish huggingface release

# 5. Comprobación final de seguridad de solo lectura antes de publicar de verdad.
#    --hfh-output-dir también hace falta aquí: es como este comando localiza
#    hfh_publication_report.json, que escribe 'huggingface release' (paso 4) dentro
#    de ese mismo directorio.
donadataset publish zenodo check-readiness --repo-id <tu-usuario>/<dataset-slug> --hfh-output-dir <export-dir>

# 6. Publicar el draft de Zenodo — IRREVERSIBLE, hazlo solo cuando el paso 5 pase
donadataset publish zenodo release --repo-id <tu-usuario>/<dataset-slug>
```

O ejecuta los pasos 1, 2, 3, 5 y 6 de una vez con `pipeline` (el paso 4, hacer público
HuggingFace Hub, es el único paso manual que queda — hazlo cuando te venga bien, antes
de ejecutar `check-readiness`):

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
- Hace la re-subida a HuggingFace Hub dentro del paso 3 (`sync-doi`) por ti
  automáticamente, en vez de dejarlo como un paso manual que tienes que recordar.
- Te avisa antes de tocar `production` si `zenodo.environment` no es `sandbox`.
- Pide confirmación explícita antes de la publicación final — el único paso
  irreversible.
- Si algún paso falla (fallo de red, error transitorio de la API...), te deja
  reintentarlo o abortar en vez de simplemente salir.

No confundir con `donadataset publish zenodo config wizard`, que solo edita campos de
`settings.toml` y no publica nada.

Ninguno de estos comandos tiene flag `--config`: siempre renderizan la única plantilla
Jinja2 incluida (`templates/Zenodo.yaml.j2`) usando `--repo-id`/`--output-dir` (y
`--hfh-output-dir` para `sync-doi`), cuyos valores por defecto vienen de
`settings.toml` — configúralos una vez con `donadataset publish zenodo config set
<campo>=...` (o `config wizard`) y evita repetir `--environment`/etc. en cada
ejecución. Todos los comandos aceptan `--dry-run` (excepto `download`, que no existe
como paso de publicación — usa `donadataset publish zenodo download` para *recuperar*
un registro ya publicado, y `pipeline`, que siempre se ejecuta de verdad). El *nombre
de la variable de entorno* del token está fijado a `ZENODO_TOKEN` en la plantilla (no
es un setting ni un flag) — pero su *valor* no hace falta exportarlo en cada sesión:
`export ZENODO_TOKEN=...` siempre gana si está definida, si no recurre a
`zenodo.token` en `settings.toml`, guardado con `donadataset publish zenodo config set
token` (entrada oculta, nunca mostrada ni por `config show`).

> ⚠️ **`zenodo release` es el único paso que publica de verdad.** Todo lo anterior
> (`prepare`, `upload`, `sync-doi`, `check-readiness`) solo crea/actualiza un
> **draft** — puedes volver a ejecutarlos tantas veces como necesites. Una vez que
> `release` tiene éxito, el registro de Zenodo y sus ficheros ya no se pueden editar
> ni eliminar.
