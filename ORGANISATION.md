# Organisation de fireviewer-evidence-ingestion

Source canonique : **[fireviewer/fireviewer-evidence-ingestion](https://github.com/fireviewer/fireviewer-evidence-ingestion)**. Responsabilité technique : **FV**. Accès : **public**.

Sources, médias, transcription et acquisition/normalisation satellite.

## Où travailler et commiter

Dans le workspace organisé, ouvrir **`actifs/fireviewer/fireviewer-evidence-ingestion`**. Chaque dossier est un dépôt Git autonome. Créer une branche de chantier dans ce dépôt ; pour un travail parallèle, créer un worktree sous `travaux/<chantier>/<composant>`. Ne jamais commiter depuis la racine multi-dépôts, `archives/` ou un ancien chemin de compatibilité.

Connecteurs et normalisation traçable des entrées ; corpus et médias restent externes.

Aucune décision de publication ni seconde base métier.

Avant un commit : vérifier `git rev-parse --show-toplevel`, `git remote -v`, `git status --short` puis `git diff --check`. Ajouter les fichiers nommément après revue. Les modifications de plusieurs composants donnent des commits distincts, reliés par leurs versions et contrats.

## Reprendre la vérification

Version de package déclarée dans les sources : **0.1.2**. Version candidate de la migration par révisions. Les tags et artefacts reçus restent immuables ; un commit de cette branche ne déploie pas le service.

```text
python tools/ci.py verify
```

Suivre les prérequis et verrous du dépôt. Les packages versionnés sont téléchargés depuis leurs releases de référence ; les ressources restant privées nécessitent leurs accès propres ; aucun dossier source voisin ne doit être nécessaire. Les secrets et fichiers `.env` réels, données, poids, corpus et sorties restent hors Git.

## Droits, historique et limites

Les licences, auteurs et notices historiques sont conservés. La responsabilité technique UWD/FV et la visibilité GitHub ne constituent pas une cession signée. Les droits tiers restent distincts.

Les anciens dépôts `fireviewer-spatial`, `fireviewer-sdg` et `models` sont remplacés et conservés en archives restaurables hors de l’organisation active. Les anciens imports nécessaires restent dans les adaptateurs explicitement maintenus. Les références historiques dans les notices et reçus gardent leur date.

La généralisation agentique des périmètres/photos, les recettes fournisseurs/modèles et Unreal natif auparavant inachevées gardent leurs propres critères de réception. Aucun succès CPU ne les déclare réalisés.

[README](README.md) · [Organisation générale](https://github.com/fireviewer/Fireviewer_doc/blob/main/docs/public/ORGANISATION.md) · [Installation et qualification](CI.md)
