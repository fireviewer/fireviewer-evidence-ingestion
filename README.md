# fireviewer-evidence-ingestion

> **Source active FV · private.** Sources, médias, transcription et acquisition/normalisation satellite. Voir [où travailler, quoi commiter et comment reprendre](ORGANISATION.md).

Acquisition and normalization of sources, media and satellite observations.

Python package: `fireviewer_evidence_ingestion`. Version: `0.1.1`.

## Installation

Install the versioned release wheels (including private FireViewer dependencies) from the release bundle. No sibling source checkout is required.

```sh
python -m pip install --find-links /path/to/release/wheels fireviewer-evidence-ingestion==0.1.1
python -m pytest tests -q
```

Optional model/provider environments are separate extras and retain their existing upstream constraints. Model weights, credentials, datasets and local evidence are external inputs.

## Ownership and compatibility

Target account: `fireviewer`. Target stewardship: Association FIRE-VIEWER. Historical authorship and AGPL-3.0-or-later notices are retained. This technical extraction is not a signed assignment of rights.

Source correspondence and hashes are recorded in the migration dossier. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. The former `firewarning_worker` or backend module paths are compatibility adapters in their original repository.

## Delivery boundary

Docker was deferred during the initial source delivery. The resumed private container phase, pinned images and acceptance limits are documented in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker). Production deployment remains separate. CPU/schema tests do not qualify GPU, visual or scientific performance.

## Sources et commandes propres au composant

Commande CPU : `fireviewer-ingestion`. Sources, médias, transcription et satellite se trouvent dans `mvp/research` et les modules satellite. Configurer les fournisseurs et leurs credentials externes explicitement avant un travail réel.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels privés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les tests de composant et leurs dépendances de test sont recensés dans le dossier unique de migration.
