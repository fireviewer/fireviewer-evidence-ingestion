# Vérification de fireviewer-evidence-ingestion 0.1.0

Depuis un checkout propre, avec Python 3.13 et le dossier `wheels` du bundle privé de migration :

```powershell
uv venv .venv --python 3.13
uv pip sync --python .venv/Scripts/python.exe --require-hashes --find-links C:/chemin/bundle/wheels --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match requirements.tests.lock.txt
.venv/Scripts/python -I -m pytest tests -q
```

Remplacer `C:/chemin/bundle/wheels` par le dossier d'artefacts livré, sans dépôt source voisin. Les hashes verrouillent les dépendances publiques et privées. Le lock contient le wheel testé de ce composant ; reconstruire un changement sous une nouvelle version avant sa réception.

`requirements.lock.txt` verrouille les dépendances de base. Le lock de test ajoute seulement les environnements nécessaires à cette suite. Torch, lorsque requis, est la distribution CPU ; aucun modèle, corpus, GPU ni appel de service n'est nécessaire. La version FAISS facultative reste ignorée si elle n'est pas installée. Les contraintes propres aux modèles des extras conservent leur identité ; ces tests ne constituent pas une qualification des poids ou du matériel.
