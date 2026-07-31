# Remplissage des dates et protection de la marge

Le fichier `meta_price_optimizer.py` ajoute une couche de yield management devant
`quote_optimizer.py`.

## Objectif

- remplir les dates encore vides lorsqu'elles approchent ;
- ne jamais proposer un produit indisponible ;
- ne jamais descendre sous la marge directe minimale définie dans `config.json` ;
- conserver la validation humaine pendant le pilote.

## Données attendues du CRM

Le CRM doit calculer le nombre de réservations confirmées et le stock encore libre
à la date de l'événement. Les simples devis non signés ne doivent pas bloquer une
machine.

```json
{
  "event_date": "2026-09-12",
  "bookings_on_date": 0,
  "date_booking_status": "empty",
  "available_products": {
    "PHOTOBOOTH": 3,
    "MIROIRBOOTH": 2,
    "BOOTH_360": 3,
    "VOGUEBOOTH": 2
  }
}
```

La quantité disponible doit être calculée ainsi :

```text
stock disponible = stock total - machines des réservations confirmées
```

## Stratégie automatique

| Situation | Objectif appliqué |
|---|---|
| Date vide à 45 jours ou moins | `fill_date` |
| Date vide à plus de 45 jours | `balanced` |
| Date presque complète | `maximize_expected_margin` |
| Produit demandé indisponible | aucun prix, alternatives disponibles |

En mode `fill_date`, le méta-modèle sélectionne le prix ayant la meilleure
probabilité de signature parmi les candidats qui respectent déjà :

- la marge directe minimale de 150 € ;
- la probabilité minimale de 60 % ;
- le plancher du catalogue ;
- la limitation de hausse lorsque le signal prix n'est pas fiable.

La priorité peut être imposée dans une demande avec :

```json
{
  "commercial_priority": "fill_date"
}
```

## Exécution

```bash
python meta_price_optimizer.py \
  --request sample_date_fill_request.json \
  --model models/signature_model.joblib \
  --output recommendation.json
```

## Résultats supplémentaires

La réponse contient notamment :

- `pricing_objective` ;
- `fill_date_mode` ;
- `date_booking_status` ;
- `bookings_on_date` ;
- `days_before_event` ;
- `requested_products_available` ;
- `available_products` ;
- `unavailable_requested_products` ;
- `alternative_products`.

Si un produit demandé n'est pas disponible, `recommendation_status` vaut
`PRODUCT_UNAVAILABLE` et `recommended_price_eur` vaut `null`.
