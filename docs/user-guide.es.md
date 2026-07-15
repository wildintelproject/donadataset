# Guía de usuario

Esta guía es para cualquiera que quiera **usar** DonaDataset — descargarlo y entrenar un
modelo con él. Si buscas publicar o actualizar el propio dataset, consulta en su lugar la
[Guía de publicación](publishing-guide.md).

---

## 1. Descargar el dataset

DonaDataset se publica en varios repositorios públicos — consulta la
[Guía de publicación](publishing-guide.md) para ver la lista completa. Esta guía cubre
cómo obtenerlo del **repositorio principal, HuggingFace Hub**, donde se publica como
fragmentos (`shards`) `.tar` (`data/<split>/*.tar`, uno o más por partición) en vez de
ficheros sueltos, para mantener manejable el número de objetos individuales del
repositorio. No necesitas lidiar con eso directamente — elige uno de los métodos de
abajo.

### La forma fácil: `scripts/download.py`

```bash
# Configurar el entorno una vez
./setup.sh
source .venv/bin/activate

# Descargar todo en ./data/
python scripts/download.py

# O solo una partición
python scripts/download.py --split train

# O en una ubicación personalizada
python scripts/download.py --output /path/to/data
```

Esto descarga los shards de la(s) partición(es) que pidas y los extrae por ti, dejando
una estructura simple `images/<split>/` + `labels/<split>/` bajo `--output`
(por defecto `./data`) — consulta la [sección 2](#2-que-te-vas-a-encontrar) para ver
exactamente cómo es. No queda nada más que los ficheros extraídos; los propios shards
`.tar` descargados se eliminan al terminar la extracción.

### La forma manual: CLI / Python de HuggingFace

Si prefieres no usar el script (por ejemplo, si solo quieres el archivo en bruto, o
estás descargando desde un fork con un `--repo-id` distinto):

```bash
pip install huggingface-hub

huggingface-cli download wildintelproject/donadataset \
  --repo-type dataset --local-dir ./donadataset-raw
```

Esto te da el repositorio completo tal como se publica — los shards `.tar` bajo `data/`,
más `README.md`, `LICENSE`, `CITATION.cff`, y los ficheros de metadatos/manifests
descritos en la
[Guía de publicación](publishing-huggingface.md#4-como-lo-subimos-cada-fichero-explicado).
Extrae tú mismo los shards de las particiones que quieras:

```bash
mkdir -p data
for shard in donadataset-raw/data/train/*.tar; do
  tar -xf "$shard" -C data
done
```

Las rutas internas de cada shard ya empiezan por `images/<split>/...` y
`labels/<split>/...`, así que extraer directamente en `data/` reconstruye la misma
estructura que produce automáticamente `scripts/download.py`.

### La forma del navegador: sin instalar nada

También puedes obtener los datos directamente desde HuggingFace Hub sin instalar nada —
navega a
[huggingface.co/datasets/wildintelproject/donadataset/tree/main](https://huggingface.co/datasets/wildintelproject/donadataset/tree/main)
y elige entre:

- Entrar en `data/<split>/` y descargar cada shard `.tar` individualmente (no hay un
  único botón de "descargar todo" en la página, así que esto solo es cómodo para unos
  pocos shards), o
- Usar el botón **"Clone repository"** de la página (o
  `git clone https://huggingface.co/datasets/wildintelproject/donadataset`, que necesita
  tener instalado [git-lfs](https://git-lfs.com/)) para obtener el repositorio completo,
  shards incluidos, de una vez.

En ambos casos acabas con los mismos shards `.tar` que con los otros dos métodos —
extráelos tú mismo exactamente como se muestra en
["La forma manual"](#la-forma-manual-cli-python-de-huggingface) arriba.

!!! note "No confundir con `donadataset publish huggingface download`"
    Ese comando pertenece a la [Guía de publicación](publishing-huggingface.md) — es una
    herramienta para mantenedores que vuelve a descargar todo el repositorio para
    verificar checksums tras una subida, requiere un `HF_TOKEN`, y no extrae nada para
    entrenamiento. Si solo quieres los datos, usa uno de los métodos de arriba.

## 2. Qué te vas a encontrar

### Estructura de directorios

```
data/
├── train/
│   ├── images/   ← imágenes de cámaras trampa (.jpg)
│   └── labels/   ← anotaciones YOLO (.txt)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Consulta [Particiones del dataset](dataset-description.md#particiones-del-dataset) para
saber para qué sirve cada partición y cómo se construyeron.

### Formato de anotación

Cada fichero de etiquetas contiene una fila por cada animal detectado:

```
<class_id> <x_center> <y_center> <width> <height>
```

Todas las coordenadas están normalizadas a `[0, 1]` relativas a las dimensiones de la
imagen. Consulta [Clases](dataset-description.md#clases) para ver a qué corresponde cada
`class_id`.

### Ficheros de metadatos

También incluidos en este repositorio (no forman parte del `data/` descargado — viven
junto a él en el checkout del proyecto):

| Fichero | Descripción |
|---------|-------------|
| `metadata/classes.yaml` | Mapea los ids de clase a nombres de especie comunes y científicos |
| `metadata/dataset.yaml` | Configuración YOLO de Ultralytics — úsala directamente con `donanet train` |

### Entrenar con él

`metadata/dataset.yaml` es una configuración de dataset YOLO de
[Ultralytics](https://docs.ultralytics.com/datasets/detect/) lista para usar, así que
cualquier configuración de entrenamiento YOLO compatible con Ultralytics puede usarla
directamente una vez hayas descargado los datos en el `path` que espera (`./data` por
defecto).

Con una instalación genérica de Ultralytics YOLO (`pip install ultralytics`):

```bash
yolo detect train data=metadata/dataset.yaml model=yolov8n.pt epochs=100
```

O con [DonaNet](https://github.com/wildintelproject/donanet), el modelo para el que se
diseñó específicamente este dataset:

```bash
donanet train --data metadata/dataset.yaml
```
