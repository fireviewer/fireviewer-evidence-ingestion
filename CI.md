# Contrôles et publication

`python tools/ci.py verify` construit le composant puis teste son installation dans un environnement isolé, sans dépôt voisin. Python 3.13, Node 24.14 et uv 0.11.7 sont fixés dans le workflow. Les dépendances Python ont des hashes; npm utilise ses lockfiles.

Les dépendances privées sont des copies exactes d'artefacts versionnés, avec leurs licences, conservées dans une release privée du consommateur et référencées par `ci.json`. Le `GITHUB_TOKEN` du dépôt suffit; aucune copie de source divergente ni aucun secret personnel ne sont ajoutés. La mise à jour d'une dépendance exige un nouveau snapshot et ses hashes.

Les pushes de branche et pull requests installent, construisent et testent. Un tag `v<version>` publie uniquement les artefacts testés du commit, présent sur `main`. Une release existante ne peut pas être écrasée. Aucun déploiement de service ne fait partie de ce workflow.

Les tests CPU ne qualifient ni les poids des modèles, ni le GPU, ni l'éditeur Unreal. Les résultats et le manifeste de release sont conservés comme artefacts Actions.
