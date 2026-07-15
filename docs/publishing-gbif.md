# Publishing to GBIF

This guide explains how DonaDataset is converted for publication to **GBIF** using the
`donadataset` CLI. It is aimed at the **dataset maintainer**.

---

## 1. What is GBIF

[GBIF](https://www.gbif.org) (Global Biodiversity Information Facility) is the world's
largest open-access aggregator of biodiversity data. It indexes hundreds of millions of
species occurrence records from research institutions, natural history museums, and
citizen science projects, and assigns a permanent **DOI** to every published dataset.
GBIF is the reference platform for ecologists, conservation biologists, and
environmental policy makers.

## 2. What format we publish in

DonaDataset publishes to GBIF as **[Camtrap DP](https://camtrap-dp.tdwg.org/)** — the
TDWG/GBIF standard for camera-trap data, natively supported by IPT v3+. It's a
Frictionless Data Package: a `datapackage.json` descriptor plus three CSV tables —
`deployments.csv` (camera placements), `media.csv` (one row per photo), and
`observations.csv` (one row per detection). GBIF ingests it by converting it internally
to Darwin Core Occurrence records.

GBIF doesn't store the images themselves — those live on
[HuggingFace Hub](publishing-huggingface.md). `media.filePath` always points there: the
persistent URL of the actual `.tar` shard (`huggingface prepare`'s output) each image was
packed into — see section 5's `media.csv` entry for exactly what that URL points at (it's
not one URL per photo).

## 3. Where the source dataset comes from

Building the Camtrap DP package means reading every image's EXIF date and counting every
label's boxes — unlike the small evidence files Zenodo re-publishes, that data only
exists inside the full dataset itself, so `prepare` needs the actual images/labels on
disk, not just a manifest — but it also needs `manifest.csv` itself, since
`media.filePath` always links to HuggingFace Hub (see section 5) and that's the only
record of which `.tar` shard packed which image.

By default (no `--source-dataset-dir`), it gets them from HuggingFace Hub
(`--hf-repo-id`, same default as `huggingface prepare`'s `repo_id`):

1. Reuses `--output-dir/hfh_download/` as-is if a previous run already extracted it
   there — no download, no network call, no re-extraction.
2. Otherwise reuses the local export at `<Documents>/donadataset/HFH/<repo_id>` if
   `huggingface prepare` already left one there — still no download.
3. Otherwise downloads the published repo (including the `.tar` shards) straight into
   `--output-dir/hfh_download/` (default `<Documents>/donadataset/GBIF/<repo_id>/
   hfh_download/`).
4. Either way, the `.tar` shards get extracted into that same `hfh_download/` directory
   (in place, if that's also where they were just downloaded).

`hfh_download/` lives *inside* `--output-dir` — `--overwrite` only ever clears the
Camtrap DP deliverables themselves (the three CSVs, `datapackage.json`, the `.zip`), never
this directory, so it survives across runs regardless of `--overwrite`.

Pass `--source-dataset-dir` to skip this resolution and point at a local YOLO folder
directly instead — it still has to be (or contain a copy of) a real `huggingface prepare`
output, `manifest.csv` included, since `--hf-repo-id` is still required and
`media.filePath` still always links to HuggingFace Hub; a raw `generate real` output
without ever having run `huggingface prepare` won't work here.

## 4. What `gbif prepare` invents, and why

This pipeline tracks no per-camera GPS coordinates or deployment dates anywhere (not in
filenames, not in a manifest) — the only per-image date info it might have is EXIF. So
`prepare` doesn't ask you to fill anything in by hand; it makes reasonable assumptions
instead, all clearly flagged in the output:

- **One deployment per split** (`train`, `val`, `test`) — treating an entire split as a
  single "camera placement" is obviously not literally true, but it's the only grouping
  this pipeline has. Each gets a distinct illustrative point inside Doñana National Park
  (see `SPLIT_DEPLOYMENT_COORDINATES` in `donadataset/services/gbif.py`) — **not** real
  camera GPS.
- **`media.timestamp`** is read from each image's EXIF `DateTimeOriginal`/`DateTime` tag
  when present. Images without a readable EXIF date (the bundled example dataset has
  none) get a timestamp interpolated within that same deployment's real EXIF range, or —
  if *no* image in the split has EXIF at all — spread across a fixed placeholder year
  (2023). Every estimated row is flagged in `media.mediaComments`
  (`"timestamp estimated: no EXIF datetime found in the source image"`); real EXIF rows
  leave that column blank.
- **`deploymentStart`/`deploymentEnd`** are the min/max of that split's *resolved*
  timestamps (real EXIF if any exists, otherwise the placeholder year) —
  `deployments.deploymentComments` notes when a deployment's range includes an estimate.

Everything else is either derived straight from the dataset (species from the YOLO
class names, per-image/per-species counts from the label boxes) or comes from settings
you already control (license, contact, institution — section 7).
`classificationMethod` is always `human` and `classifiedBy` is always `WildINTEL
experts` — fixed, not configurable, since every label in this dataset was produced by
human review rather than an automated model.

## 5. How we generate it — every file, explained

Everything below is written into `--output-dir` (default:
`<Documents>/donadataset/GBIF/<repo_id>`, or `<Documents>/donadataset/GBIF` if
`huggingface.repo_id` isn't configured yet).

### `deployments.csv`

One row per split present in the data: `deploymentID`/`locationID` (the split name),
`locationName`, `latitude`/`longitude`, `deploymentStart`/`deploymentEnd`,
`deploymentComments`.

### `media.csv`

One row per image: `mediaID` (the image id), `deploymentID`, `captureMethod`
(`activityDetection`), `timestamp`, `filePath`/`fileName`, `fileMediatype` (from the file
extension), `mediaComments`.

`filePath` always points at the actual **`.tar` shard** (`huggingface prepare`'s output,
`data/<split>/<split>-NNNNN.tar`) that image was packed into — never a local relative
path, since GBIF has no notion of "download this one from HuggingFace" and there's no
opt-out flag. `donadataset` reads `manifest.csv` from wherever the source dataset was
resolved from (section 3) — locally, no network fetch needed — to build the
`image_id → shard` mapping. This is **not** a URL per photo: every image inside the same
shard shares the same `filePath` (train images point at the train shard, val at the val
shard, and so on — a shard can also be split across several `.tar` files if the split is
large, in which case images in that split point at whichever shard they actually ended up
in). `mediaComments` always spells this out (`"filePath points to the .tar shard
containing <file>.jpg on HuggingFace Hub, not an individually downloadable file"`) so
nobody mistakes it for a direct image link. This requires the source dataset to have a
`manifest.csv` for the **same** images being scanned — a mismatch (an image `prepare`
sees locally but that isn't in the manifest) fails loudly rather than silently guessing.

`fileName` is the path *inside* that shard (e.g. `images/train/img_001.jpg`, matching the
`arcname` `huggingface prepare` gave it when packing the `.tar`) — not just the bare
filename — since that's what's actually needed to locate the image once someone
downloads and extracts the shard `filePath` points at.

### `observations.csv`

One row per image + species with at least one box (`count` = number of boxes of that
species in that image — a photo with 3 boxes of the same species is **one** row with
`count=3`, not three near-duplicates). Images whose only label is the source dataset's
`Empty` class (or that have no boxes at all) get a single `observationType=blank` row
instead of being silently dropped. Fixed columns: `observationID`, `deploymentID`,
`mediaID`, `eventID` (= `mediaID`, one photo is one event), `eventStart`/`eventEnd`,
`observationLevel` (`media`), `observationType` (`animal`/`blank`), `scientificName`,
`count`, `classificationMethod` (always `human`), `classifiedBy` (always `WildINTEL
experts`).

### `datapackage.json`

The Frictionless descriptor: title/description/license/contributors (from `gbif`
settings), `project` (sampling design, capture method), `spatial`/`temporal` coverage
derived from the deployments, `taxonomic` (every distinct species observed), and a
`resources` array describing the three CSVs (a minimal inline schema — field names only,
not the full constrained official Camtrap DP table schema).

### `<dataset-slug>-camtrap-dp.zip`

`<dataset-slug>` isn't its own flag — it's the dataset segment of `--hf-repo-id`
(`user_or_org/dataset` → `dataset`), the actual identity of what's being packaged.
Falls back to `HUGGINGFACE.dataset_slug` only when no repo_id is available at all (e.g.
`--source-dataset-dir` used without any repo_id configured).

The four files above, zipped together — this is the single file you upload to an IPT or
host yourself for `gbif register`. Run `donadataset publish gbif upload` afterward (see
section 6b) to push it as one extra file to the already-published HuggingFace Hub
dataset repo, so you get a persistent URL
(`https://huggingface.co/datasets/<repo_id>/resolve/main/<slug>-camtrap-dp.zip`) without
hosting it anywhere else. `upload` copies the `.zip` into a local HuggingFace Hub export
and regenerates that export's `checksums-sha256.txt` before pushing — both files, scoped,
so nothing else already published gets re-uploaded.

`upload` resolves that local export the same "local first, download-and-cache otherwise"
way `prepare` resolves its source dataset (section 3): it reuses
`<Documents>/donadataset/HFH/<repo_id>` (the `huggingface prepare`/`upload` output) if
it's already there, otherwise it downloads the published repo into
`<Documents>/donadataset/GBIF/<repo_id>/hfh_download` (the same cache directory `prepare`
itself uses) and reuses that cache on later runs. Pass `--hfh-output-dir` to skip this
resolution and point at a specific directory instead. `--dry-run` never downloads — if
neither location has the export yet, it just reports that it would. This needs
`HF_TOKEN` with write access and the repo to already exist (`huggingface prepare` +
`upload` already run).

## 6. Publishing — two ways to get the package into GBIF

### First-time setup

1. Create an account at [gbif.org](https://www.gbif.org) and request an
   **organisation** account for WildINTEL (or use the University of Huelva's existing
   GBIF node).
2. Either install the [GBIF IPT](https://www.gbif.org/ipt) v3+ (or use a hosted
   instance) if publishing manually (6a below), or register an **installation** (any
   type — doesn't have to be an IPT) if publishing via `gbif register`'s Registry API
   path (6b below).

### 6a. Through an IPT (manual)

1. Run `donadataset publish gbif prepare`.
2. Open your **IPT v3+** (earlier versions don't support Camtrap DP), create/update a
   resource, and upload `<dataset-slug>-camtrap-dp.zip` as its source.
3. Publish the resource from the IPT UI. GBIF indexes it within 24–48 hours and assigns
   a DOI.

### 6b. Through the Registry API (scripted, no IPT)

The IPT itself has no upload API (a
[community request for one](https://github.com/gbif/ipt/issues/1249) was closed
`Won't-fix`), but GBIF's separate Registry API lets you register a dataset and point it
at an archive you host yourself. The easiest host is the HuggingFace Hub repo you've
already published to:

```bash
donadataset publish gbif prepare

export HF_TOKEN=your-hf-write-token
donadataset publish gbif upload
# ↑ prints the persistent URL: https://huggingface.co/datasets/<repo_id>/resolve/main/donadataset-camtrap-dp.zip

export GBIF_USERNAME=your-gbif-org-username
export GBIF_PASSWORD=your-gbif-org-password
donadataset publish gbif register --archive-url https://huggingface.co/datasets/<repo_id>/resolve/main/donadataset-camtrap-dp.zip
```

(Or host `<dataset-slug>-camtrap-dp.zip` anywhere else you like and skip `gbif upload` —
`register` only cares that `--archive-url` is a public, GBIF-reachable URL.)

`donadataset publish gbif pipeline` chains all three (`prepare` -> `upload` -> `register`)
in one go.

The first run creates the dataset and adds a `CAMTRAP_DP` endpoint pointing at
`--archive-url`; it records the returned dataset UUID in
`gbif_linked_dataset_record.json` inside `--output-dir`, and every later run reads that
file, updates the dataset's metadata, and replaces the endpoint. Use
`--environment sandbox` (default) to test before `--environment production`, and
`--dry-run` to preview without calling the API.

**One-time prerequisites for 6b:** a GBIF.org account (`GBIF_USERNAME`/`GBIF_PASSWORD` —
Basic Auth, not a token), and an **organisation** + **installation** already registered
in the GBIF Registry (the installation doesn't have to be an IPT). Set their UUIDs once
with `gbif config set publishing_organization_key=...` / `installation_key=...`. The
credentials themselves don't have to be exported every session either — `GBIF_USERNAME`/
`GBIF_PASSWORD` always win if set, but otherwise fall back to `gbif.username`/
`gbif.password` in `settings.toml`, stored with `gbif config set username` / `config set
password` (hidden input, never echoed back or shown by `config show`).

## 7. Configuration

```bash
donadataset publish gbif config show
donadataset publish gbif config set contact_email=you@example.org
donadataset publish gbif config wizard
```

`institution_code` and `contact_email` (unset by default) feed `datapackage.json`'s
contributors. `environment`, `publishing_organization_key`, `installation_key`, and
`registry_language` are only used by `register` (section 6b) — `prepare` ignores them.

Deliberately **not** here: `dataset_name`, `description`, the license fields, the
contributors' organization name, and the contact's display name. Those come straight
from `HUGGINGFACE.dataset_name`/`description`/`license_id`/`license_name`/`license_url`/
`author_affiliation`/`author_family_names` — the same identity `huggingface prepare`
already uses (`donadataset publish huggingface config set ...`) — so GBIF's package
can't silently drift out of sync with what's actually published on HuggingFace Hub.
`classificationMethod`/`classifiedBy` aren't configurable either — every label in this
dataset comes from human review, so `prepare` always writes `human`/`WildINTEL experts`.

## 8. On every new version

Re-run `prepare` — it always regenerates the whole package from the current dataset, so
there's nothing to keep in sync by hand. Then either re-upload the new `.zip` to the same
IPT resource and trigger a re-crawl (6a), or re-run `register` with the same
(re-uploaded) `--archive-url` (6b) — GBIF re-crawls a changed endpoint automatically
within a few hours.
