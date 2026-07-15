# Features

## The dataset

- **8 mammal species** native to Doñana National Park (red deer, fallow deer, wild
  boar, Iberian lynx, red fox, Egyptian mongoose, European rabbit, badger) — see
  [Classes](dataset-description.md#classes).
- **YOLO-format annotations**: one label file per image, normalised bounding boxes,
  ready to train an object-detection model without any conversion step.
- **Stratified `train`/`val`/`test` splits** (70/20/10 by default), built so every
  species is represented in every split — see
  [Dataset splits](dataset-description.md#dataset-splits).
- **`.tar`-sharded storage** on HuggingFace Hub, keeping the number of individual
  objects in the repository manageable at scale, while still reconstructing a plain
  `images/<split>/` + `labels/<split>/` layout on extraction.
- **Open license** — CC BY 4.0, with citation metadata (`CITATION.cff`) generated and
  kept in sync automatically.
- **Published across multiple repositories**, each with its own DOI/PID where
  applicable (HuggingFace Hub, Zenodo, B2SHARE, GBIF) — see the
  [Publishing Guide](publishing-guide.md) for the full list and what's stored where.
- **Camtrap DP support** for GBIF: detections are exposed as a
  [Camtrap DP](https://camtrap-dp.tdwg.org/) package, making the dataset discoverable
  by ecologists and conservation researchers, not just the ML community.

## The `donadataset` CLI

- **`donadataset generate real`** — builds the clean, versioned YOLO dataset from the
  raw source data (splitting, stratified sampling, class validation).
- **`donadataset generate toy`** — carves out a small subset of an already-generated
  dataset, useful for fast local testing.
- **One publishing command per repository** — `publish huggingface`, `publish zenodo`,
  `publish b2share`, `publish gbif` — each with its own `prepare`/`upload`/`release`
  (or equivalent) steps, so every platform can be driven independently.
- **Interactive `wizard`** for HuggingFace Hub and Zenodo — walks through every phase
  step by step, asks for confirmation before irreversible actions (making a dataset
  public, publishing a Zenodo record), detects and resumes partially-completed runs,
  and lets you retry, skip, or abort a failed step instead of crashing outright.
- **Non-interactive `pipeline`** per repository, for scripted/automated runs of the
  same sequence without any prompts (used by CI or `publish all`).
- **`donadataset publish all`** — orchestrates HuggingFace Hub → Zenodo → B2SHARE →
  GBIF end to end in one command, closing the manual gaps between them automatically
  (e.g. re-uploading to HuggingFace after Zenodo reserves a DOI). Supports
  `--include`/`--exclude` to select repositories and `--dry-run` to preview the plan.
- **Built-in configuration management** — `config show`/`config set`/`config wizard`
  (global and per-integration) read and write `settings.toml`, with access tokens
  stored as real settings but always masked in `show` and entered via hidden input in
  `set`/`wizard`, never echoed to the terminal.
- **`--dry-run`** support across almost every publishing command, to preview exactly
  what would happen without touching any remote API.
- **Self-verifying exports** — every `prepare` step recomputes and checks checksums
  against its own output before anything is uploaded, and `download`/`sync-*` commands
  verify the round trip after publishing.
- **Documentation tooling** (`cli.py docs build`/`serve`/`pdf`) — builds this MkDocs
  site, serves it locally for editing, or renders it to a single PDF.
