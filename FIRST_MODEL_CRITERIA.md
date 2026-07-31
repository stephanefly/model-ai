# Critères du premier modèle de devis

Le premier modèle utilise uniquement des informations disponibles avant l'envoi du devis. Les ouvertures d'e-mail, clics, relances, réponses et paiements sont exclus.

## Critères calculés automatiquement

### Moment de la demande

- mois, jour et heure de création ;
- demande reçue le week-end ;
- demande reçue hors horaires de bureau ;
- tranche horaire : nuit, matin, après-midi ou soir.

### Délai avant l'événement

- nombre de jours avant l'événement ;
- demande très urgente : moins de 14 jours ;
- demande urgente : moins de 30 jours ;
- demande très anticipée : plus de 180 jours ;
- tranche de délai catégorielle.

Le calcul Python utilise désormais les dates civiles, comme le `DATEDIFF` de l'export SQL. Une demande reçue le soir ne perd donc plus artificiellement un jour.

### Historique antérieur du client

- nombre d'anciennes demandes ;
- nombre d'anciennes signatures ;
- taux historique de signature ;
- nombre de jours depuis la dernière demande.

Seules les demandes antérieures à la demande actuelle sont utilisées.

### Composition du devis

- nombre de machines ;
- devis avec plusieurs machines ;
- nombre de machines premium : MiroirBooth, 360 Booth et VogueBooth ;
- présence d'au moins une machine premium ;
- combinaison exacte des machines, par exemple `MIROIRBOOTH+BOOTH_360` ;
- nombre de types d'options ;
- quantité totale de porte-clés et magnets ;
- nombre total de produits et options.

### Qualité de la demande

Un score de complétude de 0 à 1 est calculé avec quatre informations :

1. adresse renseignée ;
2. horaires renseignés ;
3. commentaire du client renseigné ;
4. au moins un produit sélectionné.

Le modèle reçoit aussi le nombre d'informations manquantes.

### Prix et salle

- taux de remise par rapport au prix catalogue ;
- capacité maximale connue de la salle ;
- prix moyen estimé de la salle ;
- notes, avis, score premium et notoriété déjà présents.

## Critères prêts pour une future reconstruction historique

Les champs suivants sont autorisés par le modèle mais ne seront réellement appris que lorsqu'ils seront présents dans l'export historique :

- `bookings_on_date` ;
- `date_requested_product_bookings` ;
- `date_utilization_ratio` ;
- `available_requested_product_ratio` ;
- `event_distance_km` ;
- `event_travel_time_minutes` ;
- `event_toll_cost_eur`.

Il ne faut pas seulement envoyer ces champs pour les nouveaux devis : il faut aussi les reconstruire pour les anciens devis au moment exact où chaque devis avait été créé, sinon le modèle ne pourra pas mesurer leur effet.

## Critères volontairement exclus

- nombre d'ouvertures du devis ;
- clic sur le lien de réservation ;
- nombre de relances ;
- réponse du prospect ;
- paiement ou acompte ;
- statut final ;
- identité, origine, religion, santé ou données privées du client ;
- nombre d'invités, car les forfaits MySelfieBooth sont illimités.

## Vérification

```bash
python -m unittest tests/test_feature_engineering.py
```

Après modification des critères, il faut réentraîner le modèle et comparer le Brier score, la log-loss et les importances de variables avec la version précédente. Un critère ajouté n'est conservé que s'il améliore la validation chronologique ou apporte un garde-fou métier indispensable.

## Réentraînement obligatoire

Les nouveaux critères ne modifient pas un fichier `signature_model.joblib` déjà entraîné. Après fusion de la branche, il faut relancer :

```bash
python train_model.py \
  --data data/historique_devis_enrichi.csv \
  --output-dir models
```

Puis utiliser le nouveau fichier `models/signature_model.joblib` pour les recommandations. Les critères opérationnels absents du CSV historique seront ignorés jusqu'à ce que l'export soit enrichi.
