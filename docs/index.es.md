# DonaDataset

![WildINTEL](img/wildIntel_logo.webp){ style="display: block; margin: 0 auto;" }

**DonaDataset** es el dataset de imágenes de cámaras trampa anotadas utilizado para
entrenar [DonaNet](https://github.com/wildintelproject/donanet), una red neuronal basada
en YOLO para detectar y clasificar los mamíferos que habitan el
[Parque Nacional de Doñana](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (España).

Las imágenes y las etiquetas se publican en varios repositorios externos — consulta la
**[Guía de publicación](publishing-guide.md)** para ver la lista completa y los detalles.
Este repositorio contiene los metadatos, las definiciones de clases y los scripts
auxiliares.

---

## Mapa de la documentación

**[Descripción del dataset](dataset-description.md)**

Qué es DonaDataset, las especies de mamíferos cubiertas, cómo se recopilaron y anotaron
los datos, y las particiones (splits) del dataset.

**[Guía de usuario](user-guide.md)**

Cómo descargar el dataset y qué te vas a encontrar una vez descargado: estructura de
directorios, formato de anotación y ficheros de metadatos.

**[Guía de publicación](publishing-guide.md)** — para mantenedores

Guía paso a paso para publicar y mantener sincronizado el dataset entre HuggingFace
Hub, Zenodo, B2SHARE, GBIF y el resto de repositorios externos.

**[Características](features.md)**

Qué hace destacar a DonaDataset y a la CLI de `donadataset`: características del
dataset y capacidades de la CLI (wizards, `publish all`, gestión de configuración, y
más).

**[Acerca de](about.md)**

Contexto sobre DonaDataset, el proyecto WildINTEL y la financiación.

---

## Inicio rápido

```bash
# 1. Configurar el entorno
./setup.sh

# 2. Activar el entorno virtual
source .venv/bin/activate

# 3. Descargar todas las particiones
python scripts/download.py
```

Consulta la [Guía de usuario](user-guide.md) para ver métodos de descarga alternativos y
exactamente qué te vas a encontrar en `./data` una vez terminado.
