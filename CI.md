# Contrôles et publication

`python tools/ci.py verify` construit le composant puis teste son installation dans un environnement isolé, sans dépôt voisin. Python 3.13, Node 24.14 et uv 0.11.7 sont fixés dans le workflow. Les dépendances Python ont des hashes; npm utilise ses lockfiles.

Les dépendances FireViewer sont des copies exactes d'artefacts versionnés, avec leurs licences, conservées dans une release du consommateur et référencées par `ci.json`. Le `GITHUB_TOKEN` du dépôt suffit; aucune copie de source divergente ni aucun secret personnel ne sont ajoutés. La mise à jour d'une dépendance exige un nouveau snapshot et ses hashes.

Les pushes de branche et pull requests installent, construisent et testent. Un tag `v<version>` publie uniquement les artefacts testés du commit, présent sur `main`. Une release existante ne peut pas être écrasée. Aucun déploiement de service ne fait partie de ce workflow.

Les tests CPU ne qualifient ni les poids des modèles, ni le GPU, ni l'éditeur Unreal. Les résultats et le manifeste de release sont conservés comme artefacts Actions.

## Candidat révisions du 23 septembre 2026

Les dépendances Python nouvelles sont des wheels versionnés dans `vendor/`, avec leurs licences,
SHA-256 et origine dans `ci.json`. Les dépendances inchangées peuvent conserver leur release de
référence. Aucun artefact déjà publié n'est remplacé. Les builds utilisent les outils verrouillés et
`SOURCE_DATE_EPOCH` déclaré dans `ci.json` ; les wheels construits sont testés en installation isolée,
y compris `uv pip check`. Les branches `codex/incident-revisions-*` déclenchent les contrôles CI.
Les commandes Python nécessitent Python 3.13. Un push de branche n'est ni une release ni un déploiement.
