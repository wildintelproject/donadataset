# Descripción del dataset

## Visión general

DonaDataset es una colección de imágenes de cámaras trampa anotadas en
[formato YOLO](https://docs.ultralytics.com/datasets/detect/) para detección de objetos.
Cubre las especies de mamíferos presentes en el
[Parque Nacional de Doñana](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/)
(Huelva, España), uno de los ecosistemas de humedales más importantes de Europa. Es el
dataset de entrenamiento detrás de
[DonaNet](https://github.com/wildintelproject/donanet), una red neuronal basada en YOLO
desarrollada como parte del [proyecto WildINTEL](https://wildintel.eu/) para el
monitoreo automatizado de fauna salvaje mediante imágenes de cámaras trampa.

![Mapa de zonificación del Parque Nacional de Doñana](https://portalrediam.cica.es/geonetwork/srv/api/records/5aa36bf5092a0ac745f89d570e131f1d9deacb5e/attachments/zonificacion_donana.png)
*Mapa de zonificación del Parque Nacional de Doñana. Fuente: [REDIAM](https://portalrediam.cica.es/).*

## Clases

El dataset cubre las siguientes especies de mamíferos presentes en el Parque Nacional
de Doñana. Los ids de clase coinciden con el orden en `metadata/classes.yaml` y
`metadata/dataset.yaml`.

| ID | Nombre común       | Nombre científico       |
|----|---------------------|-------------------------|
| 0  | Ciervo rojo         | *Cervus elaphus*        |
| 1  | Gamo                | *Dama dama*             |
| 2  | Jabalí              | *Sus scrofa*            |
| 3  | Lince ibérico       | *Lynx pardinus*         |
| 4  | Zorro rojo          | *Vulpes vulpes*         |
| 5  | Meloncillo          | *Herpestes ichneumon*   |
| 6  | Conejo europeo      | *Oryctolagus cuniculus* |
| 7  | Tejón               | *Meles meles*           |

!!! note "Añadir nuevas clases"
    Para extender el dataset con nuevas especies, añade entradas a
    `metadata/classes.yaml` manteniendo los ids contiguos, actualiza
    `metadata/dataset.yaml` en consecuencia, y abre un pull request. El workflow de CI
    verificará la consistencia automáticamente.

## Protocolo de recolección

### Cámaras trampa

Las imágenes se recolectaron mediante cámaras trampa desplegadas por todo el Parque
Nacional de Doñana. Las cámaras se posicionaron en corredores de fauna conocidos, puntos
de agua y zonas de alimentación para maximizar la cobertura de especies.

### Procesamiento de imágenes

Las imágenes en bruto se exportan en formato JPEG. No se aplica corrección de color ni
redimensionado antes de la anotación — el modelo recibe las imágenes en su resolución
original.

### Flujo de anotación

Las anotaciones se produjeron usando el formato de cuadros delimitadores (bounding box)
de YOLO. Cada animal detectado se etiqueta con:

- **Id de clase** — identificador de especie (ver [Clases](#clases) arriba)
- **Cuadro delimitador** — coordenadas del centro y dimensiones normalizadas

Las detecciones ambiguas (oclusión parcial, baja confianza) se excluyen del dataset.

## Particiones del dataset

Las imágenes se dividen en particiones `train / val / test` usando el comando
`prepare-dataset` de [DonaNet](https://github.com/wildintelproject/donanet) con las
siguientes proporciones por defecto:

| Split   | Proporción | Propósito                                  |
|---------|------------|---------------------------------------------|
| `train` | 70 %       | Entrenamiento del modelo                    |
| `val`   | 20 %       | Ajuste de hiperparámetros / monitorización  |
| `test`  | 10 %       | Evaluación final (conjunto reservado)       |

Se aplica muestreo estratificado para garantizar que cada especie esté representada en
todas las particiones.

## Almacenamiento

Las imágenes y las etiquetas se publican en varios repositorios externos — consulta la
[Guía de publicación](publishing-guide.md) para ver la lista completa y los detalles.
Para saber cómo descargar realmente los datos y qué te vas a encontrar una vez
descargados (estructura de directorios, formato de anotación, ficheros de metadatos),
consulta la [Guía de usuario](user-guide.md).
