# Publicar en B2SHARE (EUDAT)

Esta guía explica cómo se publica DonaDataset en **B2SHARE** usando la CLI de
`donadataset`. Está dirigida al **mantenedor del dataset**.

---

## 1. Qué es B2SHARE

[B2SHARE](https://b2share.eudat.eu) es un servicio de intercambio de datos de
investigación operado por [EUDAT](https://eudat.eu), la infraestructura colaborativa de
datos europea. Ofrece almacenamiento seguro a largo plazo en servidores situados dentro
de la Unión Europea, y asigna un identificador persistente (un EPIC PID, y un DOI cuando
la comunidad de alojamiento lo habilita) a cada registro publicado. B2SHARE también
ofrece una instancia de **entrenamiento/sandbox** (`trng-b2share.eudat.eu`) con una API
idéntica, usada para pruebas.

Los registros en B2SHARE pertenecen a una **comunidad** — una colección curada con su
propio esquema de metadatos y (según la configuración) su propio flujo de moderación.
Necesitas el UUID de una comunidad antes de poder publicar nada; solicita uno en
b2share.eudat.eu si todavía no tienes ninguno.

## 2. Qué permite subir B2SHARE

Un registro de B2SHARE puede técnicamente contener ficheros de cualquier tipo, incluidos
datos binarios grandes — la intención original del espejo de B2SHARE de este proyecto
era una "copia europea" completa del dataset (imágenes, etiquetas y código). **En este
proyecto usamos deliberadamente en su lugar el patrón más ligero de registro enlazado**
(el mismo ya construido para [Zenodo](publishing-zenodo.md)): B2SHARE aloja solo los
metadatos y la evidencia que prueban que el dataset existe, está verificado, y dónde
encontrarlo, mientras que las imágenes/etiquetas reales permanecen en HuggingFace Hub
(ver la [guía de HuggingFace](publishing-huggingface.md)). Esto evita por completo la
cuota de almacenamiento de 10 GB por registro que tiene B2SHARE por defecto y mantiene
la publicación rápida, al coste de no tener una copia independiente completa alojada en
la UE de los datos binarios.

Esto se refuerza mediante `alternate_identifier` en los metadatos del registro de
B2SHARE: un enlace explícito a la página del dataset en HuggingFace, para que cualquiera
que llegue a la página del PID/DOI vea inmediatamente "los datos viven aquí".

## 3. Qué subimos

`donadataset publish b2share prepare` **no** necesita una copia local de la exportación
de HuggingFace para obtener los ficheros de evidencia — mismo diseño que
`zenodo prepare`. Este comando:

1. Se conecta a HuggingFace Hub (`--repo-id`) y **descarga el repositorio en vivo**, en
   su propio directorio (`--output-dir`).
2. Verifica esa descarga contra los mismos checksums/manifest con los que se construyó
   la exportación (checksums globales + hashes internos de los miembros `.tar`).
3. Extrae los pequeños ficheros de evidencia de esa misma copia descargada (nunca los
   shards `.tar`) y los sube a un nuevo draft de B2SHARE.

A diferencia de Zenodo, **B2SHARE no reserva un PID/DOI en el momento de crear el
draft** — solo se asigna una vez que el registro se publica de verdad (ver sección 5).
Así que "qué subimos" en la fase de `prepare` es: los ficheros de evidencia ya presentes
en el repositorio de HuggingFace (los mismos descritos en la
[guía de HuggingFace](publishing-huggingface.md#4-como-lo-subimos-cada-fichero-explicado)),
más un nuevo report generado por este paso de descarga-y-verificación, más un registro
JSON que ata todo junto — todavía sin identificador.

## 4. Cómo lo subimos — cada fichero explicado

### Ficheros de evidencia obtenidos de la descarga en vivo de HuggingFace

Los mismos ficheros descritos en la
[guía de HuggingFace](publishing-huggingface.md#4-como-lo-subimos-cada-fichero-explicado) —
`README.md`, `LICENSE`, `CITATION.cff`, `HuggingFaceHub.yaml`, `donana.yaml`,
`dataset_info.json`, `metadata.csv`, `manifest.csv`, `manifest-files-sha256.csv`,
`checksums-sha256.txt`, `validation_report.json`, `verification_report_local.json` —
obtenidos frescos de HuggingFace Hub en vez de leídos de una carpeta local previa a la
subida, de modo que lo que recibe B2SHARE coincide exactamente con lo que hay público en
ese momento.

### `verification_report_downloaded.json`

El resultado del paso de descarga-y-verificación en vivo descrito arriba — generado
fresco cada vez que se ejecuta `b2share prepare`, y subido a B2SHARE como una pieza más
de evidencia, exactamente igual que la versión propia de este fichero en Zenodo.

### `b2share_linked_dataset_record.json`

El **resultado** de crear (o publicar) el draft de B2SHARE — escrito localmente, no se
sube al propio B2SHARE: `record_id`, `pid`/`pid_url` (vacío hasta que se publica),
`record_url`, el entorno de B2SHARE (`sandbox`/`production`), y un bloque
`huggingface_verification` que resume el paso de descarga-y-verificación. Los comandos
posteriores (`check-readiness`, `release`, `sync-pid`) leen todos este fichero para
saber con qué registro están trabajando.

### `b2share_public_release_readiness_report.json`

Escrito por `check-readiness` — no forma parte de la subida inicial de `prepare`, pero
vive en el mismo directorio para tener un rastro de auditoría local completo.

## 5. Comandos para publicar

### Configuración inicial

1. Inicia sesión en [b2share.eudat.eu](https://b2share.eudat.eu) (o
   [trng-b2share.eudat.eu](https://trng-b2share.eudat.eu) para el sandbox) usando tu
   cuenta institucional u ORCID.
2. Solicita un UUID de **comunidad** para WildINTEL en b2share.eudat.eu si todavía no
   tienes uno — no tiene un valor por defecto razonable, todo comando que toque la red
   falla con un error claro hasta que esto esté configurado. Guárdalo mediante
   `donadataset publish b2share config set community_id=<uuid>`.
3. Crea un token de acceso personal y configúralo como `B2SHARE_TOKEN`, o guárdalo una
   vez mediante `donadataset publish b2share config set token`.

### Publicar una nueva versión

La publicación en B2SHARE ocurre **después** de que el dataset ya esté en HuggingFace
Hub (ver la
[guía de HuggingFace](publishing-huggingface.md#5-comandos-para-publicar)):

```bash
# 1. Crear el draft de B2SHARE, descargar+verificar HuggingFace Hub en vivo, subir
#    la evidencia. Usa trng-b2share.eudat.eu (sandbox) por defecto.
export B2SHARE_TOKEN='xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
donadataset publish b2share prepare \
  --repo-id <tu-usuario>/<dataset-slug> \
  --community-id <tu-uuid-de-comunidad-eudat>

# 2. Comprobación final de seguridad de solo lectura antes de publicar de verdad
donadataset publish b2share check-readiness --repo-id <tu-usuario>/<dataset-slug>

# 3. Enviar el draft para publicación
donadataset publish b2share release --repo-id <tu-usuario>/<dataset-slug>

# 4. Detectar el PID/DOI (una vez asignado) y escribirlo en CITATION.cff localmente
donadataset publish b2share sync-pid --repo-id <tu-usuario>/<dataset-slug>

# 5. Volver a subir a HuggingFace Hub para que se publique el CITATION.cff actualizado
donadataset publish huggingface upload
```

O ejecuta los pasos 1, 2, 3 y 4 de una vez con `pipeline`:

```bash
donadataset publish b2share pipeline --repo-id <tu-usuario>/<dataset-slug> \
  --community-id <tu-uuid-de-comunidad-eudat>
```

Ninguno de estos comandos tiene flag `--config`: siempre renderizan la única plantilla
Jinja2 incluida (`templates/B2SHARE.yaml.j2`) usando
`--repo-id`/`--output-dir`/`--community-id`, cuyos valores por defecto vienen de
`settings.toml` — configúralos una vez con `donadataset publish b2share config set
<campo>=...` (o `config wizard`) y evita repetirlos en cada ejecución. El *nombre de la
variable de entorno* del token está fijado a `B2SHARE_TOKEN` en la plantilla (no es un
setting ni un flag) — pero su *valor* no hace falta exportarlo en cada sesión: `export
B2SHARE_TOKEN=...` siempre gana si está definida, si no recurre a `b2share.token` en
`settings.toml`, guardado con `donadataset publish b2share config set token` (entrada
oculta, nunca mostrada ni por `config show`).

> ⚠️ **`community_id` no tiene un valor por defecto razonable — primero debes
> solicitar uno a EUDAT.** Hasta que configures `donadataset publish b2share config set
> community_id=<uuid>`, todo comando que toque la red fallará con un error claro que lo
> explica.

> ⚠️ **La publicación puede requerir aprobación de un moderador.** A diferencia de
> Zenodo (que genera un DOI en el momento en que publicas), las comunidades de B2SHARE
> pueden configurarse para requerir que un moderador apruebe los registros nuevos antes
> de que se asigne un PID/DOI. `b2share release` envía el registro para publicación;
> `b2share sync-pid` es el paso que más tarde detecta si ya se ha asignado un
> identificador, y solo entonces actualiza tu `CITATION.cff` local — vuelve a ejecutarlo
> más tarde si la primera vez informa de que no ha encontrado nada.
