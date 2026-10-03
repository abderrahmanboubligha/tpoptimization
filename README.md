# tpoptimization

TP d'optimisation avec NumPy - BOUBLIGHA ABDERRAHMAN.

Rapport final : [rapport_BOUBLIGHA_ABDERRAHMAN.pdf](rapport_BOUBLIGHA_ABDERRAHMAN.pdf).

## Exécution

Installer Python 3.12 ou plus récent, puis ouvrir un terminal dans ce dossier :

```text
python -m pip install -r requirements.txt
python tp.py --epochs 150
```

Le programme vérifie les gradients avant de lancer les expériences. Une erreur supérieure à 0,001 arrête le programme. Les résultats sont enregistrés dans `resultats/` : tableaux CSV, courbes par exécution, figures et versions.

Les modèles, gradients et les sept optimiseurs sont programmés avec NumPy. Scikit-learn sert seulement aux métriques, jamais à entraîner un modèle.

## Organisation

- `tp.py` : code avec fonctions simples et commentaires.
- `donnees/` : les trois CSV de l'archive fournie et sa description.
- `rapport.tex` : rapport autonome prévu pour 5 pages ; ouvrir dans un éditeur LaTeX ou Overleaf.
- `rapport_BOUBLIGHA_ABDERRAHMAN.pdf` : rapport final de 5 pages avec le nom de l'étudiant et les figures détaillées. Il a été généré séparément ; la source LaTeX contient des courbes de comparaison à la place des figures externes. La compilation LaTeX n'a pas pu être vérifiée : le compilateur intégré échoue au démarrage avec « Unable to find standard directories for platform ».
- `resultats/resultats_bruts.csv` : chaque exécution et ses coûts.
- `resultats/resume.csv` : moyenne et écart type sur les graines 42, 123, 2024.
- `resultats/gradients.csv` : vérification par différences finies.

## Protocole

150 époques pour les CSV, lots de 32, deux pas candidats par méthode. Le pas est choisi sur la validation pour régression/classification et sur l'objectif sans étiquette pour clustering. Le choix est effectué avec la graine 42, puis conservé pour les trois graines. Chaque graine utilise une initialisation identique entre méthodes. Le meilleur état selon le critère de sélection est conservé. Les temps et compteurs des tableaux concernent l'entraînement final, sans inclure les essais de sélection ni les figures ; les courbes de tous les essais non retenus ne sont pas enregistrées.

OLS : lambda = 0. Ridge/RBF : lambda parmi 0,001 ; 0,01 ; 0,1. Sigma RBF parmi 0,5 ; 1 ; 2. La résolution linéaire sélectionne ces hyperparamètres sur validation et sert de contrôle ; les résultats des sept méthodes proviennent ensuite de véritables descentes. Le biais linéaire n'est pas pénalisé. La perte affichée inclut la régularisation ; la RMSE de sélection ne l'inclut pas.

Clustering : tau = 0,25, K = 3. Les classes réelles sont utilisées uniquement pour les métriques finales et la figure de composition. Lloyd est une référence distincte.

## Limites importantes

L'archive fournie ne contient pas les cinq jeux réels du sujet. Le rapport décrit seulement les trois jeux effectivement exécutés. Ce dossier est une version simplifiée : il ne satisfait pas toutes les exigences du sujet complet (huit jeux et nombreuses figures spécialisées).

Le programme détecte aussi les CSV réels s'ils sont ajoutés avec les noms et colonnes du sujet, ainsi que les fichiers NPZ MNIST/Fashion-MNIST contenant x_train, y_train, x_val, y_val, x_test, y_test. Ces branches ne sont pas validées en l'absence des données. Pour les images, il utilise les 50 000 images train, les lots 1 024/2 048 et un budget préliminaire de 8 époques. Leurs visualisations spécialisées restent à compléter avant une remise du TP complet.

La silhouette est N/A si elle n'est pas définie (un seul cluster sur l'échantillon). Les numéros de clusters sont arbitraires et ne représentent pas les numéros de classes.
