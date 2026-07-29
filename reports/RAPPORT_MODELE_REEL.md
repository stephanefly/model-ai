# Rapport du modèle réel MySelfieBooth

Date d'entraînement : 29 juillet 2026

## Verdict

| Élément | Verdict |
|---|---|
| Score de potentiel de signature | Exploitable en phase pilote |
| Comparaison au taux moyen | Meilleur |
| Validation sur les demandes récentes | Réussie |
| Optimisation automatique du prix | Pas encore totalement prouvée |
| Hausse tarifaire sans signal prix suffisant | Limitée à +10 % |
| Validation humaine | Obligatoire |

## Données utilisées

| Indicateur | Résultat |
|---|---:|
| Demandes exportées | 1 693 |
| Demandes entraînables | 1 459 |
| Signées | 699 |
| Non signées | 760 |
| Taux de signature historique | 47,9 % |
| Période | mai 2021 à juillet 2026 |
| Demandes de validation récentes | 292 |
| Taux de signature de la période récente | 36,6 % |

Le test final commence au 16 septembre 2025. Ces demandes récentes n'ont pas été
utilisées pour apprendre le modèle évalué.

## Résultats

| Mesure | Modèle retenu | Référence naïve |
|---|---:|---:|
| ROC AUC | 0,821 | 0,500 environ |
| Brier, plus bas = meilleur | 0,180 | 0,252 |
| Log-loss, plus bas = meilleur | 0,548 | 0,697 |
| Average precision | 0,790 | — |

Le modèle retenu pour le score de conversion est une forêt aléatoire calibrée.
Une régression logistique séparée est utilisée pour tester les variations de prix,
car son comportement entre deux prix est plus stable.

## Fuite de données supprimée

Le premier test donnait un résultat artificiellement excellent car le champ
`event_has_schedule` était principalement rempli après la signature. Il a été
retiré, ainsi que les commentaires internes qui peuvent évoluer après le devis.

Le modèle final n'utilise pas :

- les statuts ou l'issue du devis ;
- les paiements et acomptes ;
- les relances et ouvertures d'e-mails ;
- l'horaire final ;
- les commentaires internes ;
- les noms, e-mails ou téléphones ;
- l'adresse exacte en tant que variable directe.

## Variables les plus utiles

| Rang | Variable |
|---:|---|
| 1 | Longueur du commentaire initial du client |
| 2 | Remise globale |
| 3 | Style de mur floral demandé |
| 4 | Remises sur les options |
| 5 | Remises sur les prestations |
| 6 | Délai entre la demande et l'événement |
| 7 | Durée de la prestation |
| 8 | Rapport prix proposé / prix catalogue |
| 9 | Options et prestations sélectionnées |
| 10 | Code postal, ville et source de la demande |

## Limite actuelle sur le prix

L'historique permet de prédire le potentiel global d'une demande, mais il ne permet
pas encore de mesurer précisément ce qui se serait passé si le même client avait
reçu 690 €, 790 € ou 890 €.

Le moteur applique donc les règles suivantes :

1. prix finissant par 90 € ;
2. marge directe minimale de 150 € ;
3. probabilité minimale visée de 60 % ;
4. probabilité forcée à ne jamais augmenter avec le prix ;
5. hausse limitée à 10 % du catalogue si la baisse de probabilité n'est pas assez
   visible dans les données ;
6. validation humaine pendant le pilote.

## Qualité des données à corriger

| Problème | Nombre |
|---|---:|
| Prix proposé manquant | 98 |
| Délai événement négatif | 19 |
| Prix catalogue nul ou négatif | 7 |
| Prix proposé inférieur à 50 % du catalogue | 21 |
| Prix proposé supérieur à 150 % du catalogue | 19 |

Ces lignes ne bloquent pas le premier modèle, mais doivent être vérifiées pour les
prochains entraînements.

## Salle et adresse

1 625 demandes ont une adresse et 1 507 lieux distincts ont été détectés. Pour
l'instant, le modèle utilise la ville, le code postal et une clé anonyme du lieu.

La capacité, la note, la notoriété et le niveau premium de la salle ne sont pas
encore présents dans cet entraînement. L'étape suivante consiste à enrichir les
adresses avec Google Places et les sites officiels, puis à réentraîner le modèle.

## Mise en production recommandée

| Phase | Utilisation |
|---|---|
| 1 — Observation | Afficher le score et le prix conseillé sans modifier le devis |
| 2 — Pilote | Validation humaine systématique sur 50 à 100 demandes |
| 3 — Test contrôlé | Comparer méthode actuelle et recommandation du modèle |
| 4 — Automatisation | Seulement après amélioration démontrée de la marge par demande |

Le principal indicateur à suivre est :

`marge directe totale signée / nombre de demandes reçues`

Il faut également enregistrer chaque version du devis, le prix présenté, la date,
le résultat et le motif de refus. Ces données permettront au futur modèle de mieux
apprendre la véritable sensibilité au prix.
