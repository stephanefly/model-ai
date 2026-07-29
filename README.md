# Modèle de conversion et d'optimisation des devis MySelfieBooth

Cette V1 est indépendante du CRM. Elle apprend la probabilité de signature à partir
de l'historique, puis teste plusieurs prix finissant par **90 €**. Elle recommande
le prix qui maximise la marge directe attendue tout en imposant :

- une marge directe minimale de **150 €** ;
- une probabilité de signature minimale de **60 %** ;
- une validation humaine pendant la phase pilote ;
- une probabilité qui ne peut pas augmenter lorsque le prix augmente.
- une hausse limitée à 10 % du catalogue si l'historique n'apprend pas encore
  une sensibilité au prix suffisante.

Le nombre d'invités n'est pas requis : les prestations étant illimitées, le modèle
s'appuie surtout sur les prestations/options, le prix, la date, le délai avant
l'événement, la source de la demande, l'historique client anonymisé et le niveau de
la salle.

## 1. Données à exporter

Ouvrir `export_ml_myselfiebooth_mysql.sql` dans MySQL Workbench ou votre outil SQL.
Le script est en lecture seule. Pour produire le CSV :

1. exécuter les contrôles des sections 0 et 1 ;
2. sélectionner puis exécuter uniquement la section **2. EXPORT PRINCIPAL**,
   depuis `WITH email_stats AS (` jusqu'à `ORDER BY e.created_at ASC;` ;
3. exporter la grille obtenue en CSV UTF-8 sous le nom `historique_devis.csv`.

L'export inclut toutes les données actuelles utiles :

- demande, date et lieu ;
- produits et options ;
- prix catalogue, remises et prix proposé ;
- historique antérieur anonymisé du même client ;
- signature/refus, prix validé et informations de paiement disponibles ;
- relances et ouvertures d'e-mail pour analyse séparée.

Il exclut les noms, e-mails et téléphones. La base actuelle ne contient pas de table
`Facture` : les ventes et paiements sont représentés par `Event`, `EventAcompte` et
`EventPostPrestation`.

Les colonnes de résultat, de paiement, d'e-mail ou de relance ne servent jamais à
prédire le devis initial. Elles servent uniquement à déterminer le résultat ou à
faire des analyses après l'envoi.

## 2. Installation

Depuis le dossier du projet :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Sous Windows, l'activation est :

```powershell
.venv\Scripts\activate
```

## 3. Enrichissement des salles

Si le champ contient le nom de la salle, son adresse, ou les deux, le script cherche
le meilleur établissement correspondant. Il utilise Google Places seulement si une
clé est fournie :

```bash
export GOOGLE_PLACES_API_KEY="VOTRE_CLE"
python enrich_venues.py \
  --input data/historique_devis.csv \
  --output data/historique_devis_enrichi.csv \
  --cache data/venue_cache.json
```

Pour rechercher aussi une capacité déclarée sur le site officiel de la salle :

```bash
python enrich_venues.py \
  --input data/historique_devis.csv \
  --output data/historique_devis_enrichi.csv \
  --cache data/venue_cache.json \
  --scrape-official-websites
```

La capacité reste vide si elle n'est pas trouvée de façon crédible. Les corrections
manuelles vérifiées peuvent être ajoutées dans `venue_overrides.csv`, avec la clé
`event_venue_anonymous_key`. Par défaut, les adresses exactes temporaires sont
supprimées du CSV enrichi. Le cache évite de repayer la même recherche.

Documentation de l'API :
[Google Places Text Search](https://developers.google.com/maps/documentation/places/web-service/text-search).

## 4. Entraînement

```bash
python train_model.py \
  --data data/historique_devis_enrichi.csv \
  --output-dir models
```

Le programme :

1. conserve seulement les demandes terminées avec un prix proposé ;
2. trie les données dans le temps ;
3. entraîne une régression logistique et une forêt aléatoire calibrées ;
4. teste les deux modèles sur les 20 % de demandes les plus récentes ;
5. compare leurs probabilités avec un modèle naïf qui prédit toujours le taux moyen ;
6. enregistre le modèle, les métriques et l'importance des variables.

Le meilleur modèle sert au score de conversion. Une régression logistique séparée,
plus stable entre deux prix, sert aux simulations tarifaires.

Fichiers produits :

- `models/signature_model.joblib` : modèle exécutable ;
- `models/training_report.json` : mesures de qualité ;
- `models/feature_importance.csv` : variables réellement utiles.

Ne pas passer en production si `validation_status` n'est pas `PASS`. L'entraînement
est également bloqué avec moins de 80 dossiers exploitables ou moins de 15 signatures
ou non-signatures.

## 5. Recommandation d'un devis

Adapter `sample_new_request.json`, puis lancer :

```bash
python quote_optimizer.py \
  --request sample_new_request.json \
  --model models/signature_model.joblib \
  --output recommendation.json
```

Le résultat montre chaque prix testé, sa probabilité estimée, la marge directe et la
marge attendue :

```text
marge attendue = probabilité de signature × (prix proposé − coût direct)
```

Le prix retenu est le plus élevé parmi les candidats situés à moins de 5 % de la
meilleure marge attendue, à condition d'atteindre la probabilité minimale. Si aucun
prix ne l'atteint, le moteur choisit uniquement comme solution de secours le prix
ayant la meilleure probabilité, place `all_constraints_satisfied` à `false` et
impose une révision humaine.

Le coût direct peut être calculé depuis `catalogue_tarifs.csv` ou fourni explicitement
avec `direct_cost_eur`. Un coût manquant provoque une erreur : il n'est jamais
remplacé silencieusement par zéro.

Le moteur applique également le plancher commercial du catalogue. Pour le pack
MiroirBooth + 360 Booth, il reconnaît automatiquement `PACK_DUO`, même lorsqu'un
coût direct manuel est fourni.

Pour chaque candidat, la remise totale est recalculée automatiquement à partir du
prix brut et du prix testé. Cela permet de présenter au modèle un pack à 1 050 €
vendu 750 € comme une remise produit de 300 €, conformément aux données historiques.

## 6. Comment prouver que cela fonctionne

Une bonne validation se fait en deux étapes.

### Validation historique

- `validation_status = PASS` ;
- Brier et log-loss meilleurs que le modèle naïf ;
- probabilités cohérentes par tranche (par exemple, les devis annoncés à 70 % signent
  environ 70 % du temps) ;
- contrôle manuel des principales variables pour détecter une fuite de données ;
- aucun identifiant personnel ou résultat postérieur au devis dans les variables.

### Pilote commercial

Pendant 6 à 8 semaines, garder la validation humaine et répartir aléatoirement les
demandes comparables entre :

- groupe contrôle : méthode tarifaire actuelle ;
- groupe modèle : recommandation du moteur, sans descendre sous la marge minimale.

Comparer le taux de signature, le panier signé, la marge par demande reçue et les
annulations. Le critère principal recommandé est la **marge directe totale par
demande**, pas seulement le taux de signature.

L'historique seul ne prouve pas l'effet causal du prix, car les anciens prix ont été
choisis par des humains. Pour améliorer le modèle, le CRM devra conserver chaque
version du devis : prix affiché, date, options, acceptation/refus et motif de perte.

## 7. Sécurité et usage

- Ne pas noter l'origine, l'ethnie, la religion, la santé ou la vie privée du client.
- Limiter la recherche Web à la salle ou à une entreprise cliente.
- Afficher une information claire sur l'utilisation des données pour personnaliser
  les devis ; la bannière cookies seule ne décrit pas forcément ce traitement.
- Restreindre l'accès aux exports, fixer une durée de conservation et journaliser les
  décisions du modèle.
- Garder une validation humaine tant que le test contrôlé n'a pas démontré un gain.

## Test technique sans vraies données

Ce test vérifie l'installation, mais ne doit jamais servir à établir un vrai prix :

```bash
python generate_demo_data.py --output data/demo_history.csv
python train_model.py --data data/demo_history.csv --output-dir models-demo
python quote_optimizer.py \
  --request sample_new_request.json \
  --model models-demo/signature_model.joblib
```
