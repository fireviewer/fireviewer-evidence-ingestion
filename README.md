# fireviewer-evidence-ingestion

## Repères documentaires — 19 septembre 2026

- **Rôle :** Acquisition et normalisation bornées des sources, médias, transcription et observations satellite.
- **Statut :** Actif — package v0.1.1.
- **Entrées :** Sources autorisées/officielles/publiques et credentials fournisseurs configurés explicitement.
- **Sorties :** Évidences normalisées et métadonnées de provenance exploitables par le backend/pipeline.
- **Limites :** Ne pas aspirer sans limite ni transformer le dépôt en archive brute de médias. Droits, hash, temps et résultat d’acquisition doivent rester traçables.

[Fiche du dépôt](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/repositories/fireviewer-evidence-ingestion.md) · [Architecture](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ARCHITECTURE.md) · [Statuts et vocabulaire](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/STATUTS_ET_VOCABULAIRE.md).

Cette revue documentaire ne renouvelle aucun test ni aucune réception. Les procédures, versions et preuves techniques ci-dessous conservent leur périmètre et leur date.

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

## Canonical repository and rights

Canonical source: [`fireviewer/fireviewer-evidence-ingestion`](https://github.com/fireviewer/fireviewer-evidence-ingestion). Technical stewardship: FIRE-VIEWER. Repository access: private.

Historical authorship, AGPL-3.0-or-later notices and third-party rights are retained. Technical stewardship and repository placement are not a signed assignment of intellectual-property rights. Any pre-association assets remain subject to their documented licences or agreements.

This repository is the maintained implementation location for the responsibility stated above. Existing schema IDs, algorithm revisions and evidence/publication gates are preserved. Older `firewarning_worker` or backend imports remain compatibility adapters where required; they are not alternative locations for new component logic.

## Delivery and qualification

Versioned packages are distributed through the authorised private release bundles. Current container locks, reconstruction inputs and dated acceptance records are maintained in [fireviewer-docker](https://github.com/fireviewer/fireviewer-docker).

Package installation, CPU/schema tests, service deployment and real-data acceptance are separate checks. CPU/schema tests do not qualify GPU, visual or scientific performance. This documentation update does not publish a package, rebuild an image or change production configuration.

Extraction correspondence and hashes remain in the historical migration dossier. They record the restructuring, not the current deployment state.

## Sources et commandes propres au composant

Commande CPU : `fireviewer-ingestion`. Sources, médias, transcription et satellite se trouvent dans `mvp/research` et les modules satellite. Configurer les fournisseurs et leurs credentials externes explicitement avant un travail réel.

Les dépendances de base sont verrouillées avec hashes dans `requirements.lock.txt` (Python 3.13). Installer les wheels privés du même bundle via `--find-links`. Les extras lourds restent liés à leurs versions existantes et ne qualifient aucun GPU. Les commandes de reprise et leurs prérequis sont décrits dans [ORGANISATION.md](ORGANISATION.md). Les reçus du dossier de migration restent des preuves historiques, pas une nouvelle qualification.
