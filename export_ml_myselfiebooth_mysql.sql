/*
MySelfieBooth - Export Machine Learning pour optimisation des devis
Source du schema : stephanefly/reservation, app/models.py
Dialecte : MySQL 8 / MariaDB recente

IMPORTANT
---------
1. Ce script est 100 % en lecture seule : SELECT uniquement.
2. Il produit une ligne par demande/devis (app_event).
3. Les noms, e-mails et telephones ne sont pas exportes.
4. L'adresse exacte est exportee uniquement dans les colonnes enrichment_* afin
   d'identifier la salle sur le Web. Elle ne doit jamais etre utilisee directement
   par le modele et doit etre supprimee du jeu final apres enrichissement.
5. Les donnees posterieures a l'envoi du devis sont conservees pour creer
   les cibles et analyser les relances, mais elles ne doivent pas servir
   de variables au modele de prix initial.
6. La base ne contient pas de table Facture. Les ventes sont representees
   par app_event, app_eventacompte et app_eventpostprestation.

COLONNES A PRODUIRE PAR LE FUTUR ENRICHISSEMENT WEB
---------------------------------------------------
- venue_name
- venue_match_confidence
- venue_is_private_address
- venue_type
- venue_capacity_seated
- venue_capacity_cocktail
- venue_rating
- venue_review_count
- venue_price_min
- venue_price_max
- venue_has_official_website
- venue_premium_score
- venue_notability_score
- venue_enrichment_sources_count
*/


/* ================================================================
   0. CONTROLE RAPIDE DE LA BASE
   ================================================================ */

SELECT VERSION() AS mysql_version, DATABASE() AS database_name;

SELECT table_name
FROM information_schema.tables
WHERE table_schema = DATABASE()
  AND table_name LIKE 'app\\_%'
ORDER BY table_name;


/* ================================================================
   1. CONTROLE DE LA QUALITE DES STATUTS
   ================================================================ */

SELECT
    COALESCE(e.status, 'NULL') AS status,
    COUNT(*) AS nombre_demandes,
    SUM(
        CASE
            WHEN e.signer_at IS NOT NULL
              OR e.prix_valided IS NOT NULL
              OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media')
            THEN 1 ELSE 0
        END
    ) AS nombre_signees,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN e.signer_at IS NOT NULL
                  OR e.prix_valided IS NOT NULL
                  OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media')
                THEN 1 ELSE 0
            END
        ) / NULLIF(COUNT(*), 0),
        1
    ) AS taux_signature_pct
FROM app_event e
GROUP BY COALESCE(e.status, 'NULL')
ORDER BY nombre_demandes DESC;


/* ================================================================
   2. EXPORT PRINCIPAL POUR LE MACHINE LEARNING

   VARIABLES AUTORISEES POUR LE MODELE DE PRIX INITIAL :
   - colonnes request_*
   - colonnes event_*
   - colonnes product_*
   - colonnes option_*
   - colonnes price_* connues au moment de la proposition
   - colonnes prior_*

   COLONNES INTERDITES COMME VARIABLES DU MODELE INITIAL :
   - target_*
   - outcome_*
   - signed_*
   - payment_*
   - postpresta_*
   - email_*
   - followup_*
   Elles servent uniquement de cible, de controle ou au second modele
   de relance apres envoi.
   ================================================================ */

WITH
email_stats AS (
    SELECT
        et.event_traced AS event_id,
        COUNT(*) AS email_sent_count,
        SUM(CASE WHEN et.opened = 1 THEN 1 ELSE 0 END) AS email_open_count,
        MAX(CASE WHEN et.opened = 1 THEN 1 ELSE 0 END) AS email_any_opened,
        MIN(et.sent_at) AS email_first_sent_at,
        MAX(et.sent_at) AS email_last_sent_at,
        MIN(CASE WHEN et.opened = 1 THEN et.opened_at END) AS email_first_opened_at,
        MAX(CASE WHEN et.opened = 1 THEN et.opened_at END) AS email_last_opened_at
    FROM app_emailtracking et
    WHERE et.event_traced IS NOT NULL
    GROUP BY et.event_traced
),

followup_stats AS (
    SELECT
        er.event_id,
        COUNT(*) AS followup_count,
        MIN(er.date_relance) AS followup_first_at,
        MAX(er.date_relance) AS followup_last_at,
        MAX(COALESCE(er.qualification, 0)) AS followup_max_qualification,
        ROUND(AVG(COALESCE(er.qualification, 0)), 2) AS followup_avg_qualification,
        SUM(CASE WHEN er.commentaire IS NOT NULL AND TRIM(er.commentaire) <> '' THEN 1 ELSE 0 END)
            AS followup_with_comment_count
    FROM app_eventrelance er
    GROUP BY er.event_id
),

client_history AS (
    SELECT
        current_event.id AS event_id,
        COUNT(previous_event.id) AS prior_request_count,
        SUM(
            CASE
                WHEN previous_event.id IS NOT NULL
                 AND previous_event.signer_at IS NOT NULL
                 AND previous_event.signer_at < current_event.created_at
                THEN 1 ELSE 0
            END
        ) AS prior_signed_count,
        MAX(previous_event.created_at) AS prior_last_request_at
    FROM app_event current_event
    LEFT JOIN app_event previous_event
        ON previous_event.client_id = current_event.client_id
       AND previous_event.created_at < current_event.created_at
    GROUP BY current_event.id
),

outcomes AS (
    SELECT
        e.id AS event_id,
        CASE
            WHEN e.signer_at IS NOT NULL
              OR e.prix_valided IS NOT NULL
              OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media')
            THEN 1 ELSE 0
        END AS target_signed,
        CASE
            WHEN e.signer_at IS NOT NULL
              OR e.prix_valided IS NOT NULL
              OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media')
            THEN 'SIGNED'
            WHEN e.status = 'Refused'
            THEN 'REFUSED'
            WHEN ed.date_evenement < CURRENT_DATE()
            THEN 'EXPIRED_NO_SIGNATURE'
            ELSE 'OPEN'
        END AS outcome_class,
        CASE
            WHEN e.signer_at IS NOT NULL
              OR e.prix_valided IS NOT NULL
              OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media')
              OR e.status = 'Refused'
              OR ed.date_evenement < CURRENT_DATE()
            THEN 1 ELSE 0
        END AS target_training_eligible,
        CASE
            WHEN e.prix_proposed IS NOT NULL
             AND e.prix_brut > 0
             AND (
                    e.signer_at IS NOT NULL
                 OR e.prix_valided IS NOT NULL
                 OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media', 'Refused')
                 OR ed.date_evenement < CURRENT_DATE()
             )
            THEN 1 ELSE 0
        END AS target_price_training_eligible
    FROM app_event e
    INNER JOIN app_eventdetails ed ON ed.id = e.event_details_id
)

SELECT
    /* Identifiants techniques, sans identite client */
    e.id AS request_event_id,
    e.num_devis AS request_quote_number,
    e.created_at AS request_created_at_utc,

    /* Informations client connues au moment de la demande */
    c.how_find AS request_source,
    CASE WHEN c.raison_sociale = 1 THEN 1 ELSE 0 END AS request_is_business,

    /* Historique anterieur du meme client, sans utiliser nom/mail/telephone */
    COALESCE(ch.prior_request_count, 0) AS prior_request_count,
    COALESCE(ch.prior_signed_count, 0) AS prior_signed_count,
    ch.prior_last_request_at AS prior_last_request_at_utc,
    CASE
        WHEN COALESCE(ch.prior_request_count, 0) > 0 THEN 1 ELSE 0
    END AS prior_is_returning_client,

    /* Evenement */
    ed.date_evenement AS event_date,
    MONTH(ed.date_evenement) AS event_month,
    QUARTER(ed.date_evenement) AS event_quarter,
    WEEKDAY(ed.date_evenement) + 1 AS event_weekday_iso,
    CASE WHEN WEEKDAY(ed.date_evenement) >= 5 THEN 1 ELSE 0 END AS event_is_weekend,
    DATEDIFF(ed.date_evenement, DATE(e.created_at)) AS event_lead_time_days,
    ed.code_postal_evenement AS event_postal_code,
    LOWER(TRIM(ed.ville_evenement)) AS event_city,
    ed.adresse_evenement AS enrichment_event_address_raw,
    CONCAT_WS(
        ', ',
        NULLIF(TRIM(ed.adresse_evenement), ''),
        CONCAT(
            COALESCE(ed.code_postal_evenement, ''),
            ' ',
            COALESCE(ed.ville_evenement, '')
        )
    ) AS enrichment_event_full_address_raw,
    CASE
        WHEN ed.adresse_evenement IS NULL OR TRIM(ed.adresse_evenement) = ''
        THEN 0 ELSE 1
    END AS enrichment_address_available,
    CASE
        WHEN ed.horaire IS NULL OR TRIM(ed.horaire) = '' THEN 0 ELSE 1
    END AS event_has_schedule,
    ed.horaire AS event_schedule_raw,
    CASE
        WHEN ed.comment_client IS NULL OR TRIM(ed.comment_client) = '' THEN 0 ELSE 1
    END AS event_has_client_comment,
    CHAR_LENGTH(COALESCE(ed.comment_client, '')) AS event_client_comment_length,
    CASE
        WHEN ed.comment IS NULL OR TRIM(ed.comment) = '' THEN 0 ELSE 1
    END AS event_has_internal_comment,
    CHAR_LENGTH(COALESCE(ed.comment, '')) AS event_internal_comment_length,
    SHA2(
        CONCAT_WS(
            '|',
            LOWER(TRIM(COALESCE(ed.adresse_evenement, ''))),
            COALESCE(ed.code_postal_evenement, ''),
            LOWER(TRIM(COALESCE(ed.ville_evenement, '')))
        ),
        256
    ) AS event_venue_anonymous_key,

    /* Prestations demandees */
    COALESCE(ep.photobooth, 0) AS product_photobooth,
    COALESCE(ep.miroirbooth, 0) AS product_miroirbooth,
    COALESCE(ep.videobooth, 0) AS product_videobooth,
    COALESCE(ep.voguebooth, 0) AS product_voguebooth,
    COALESCE(ep.ipadbooth, 0) AS product_ipadbooth,
    COALESCE(ep.airbooth, 0) AS product_airbooth,
    (
          COALESCE(ep.photobooth, 0)
        + COALESCE(ep.miroirbooth, 0)
        + COALESCE(ep.videobooth, 0)
        + COALESCE(ep.voguebooth, 0)
        + COALESCE(ep.ipadbooth, 0)
        + COALESCE(ep.airbooth, 0)
    ) AS product_count,

    /* Options demandees */
    COALESCE(eo.`MurFloral`, 0) AS option_mur_floral,
    eo.mur_floral_style AS option_mur_floral_style,
    COALESCE(eo.`Phonebooth`, 0) AS option_phonebooth,
    COALESCE(eo.`LivreOr`, 0) AS option_livre_or,
    COALESCE(eo.`Fond360`, 0) AS option_fond_360,
    COALESCE(eo.`PanneauBienvenue`, 0) AS option_panneau_bienvenue,
    COALESCE(eo.`PhotographeVoguebooth`, 0) AS option_photographe_voguebooth,
    COALESCE(eo.`ImpressionVoguebooth`, 0) AS option_impression_voguebooth,
    COALESCE(eo.`DecorVoguebooth`, 0) AS option_decor_voguebooth,
    COALESCE(eo.`Holo3D`, 0) AS option_holo_3d,
    COALESCE(eo.`PanneauFontaine`, 0) AS option_panneau_fontaine,
    COALESCE(eo.`VideoLivreOr`, 0) AS option_video_livre_or,
    COALESCE(eo.magnets, 0) AS option_magnets_quantity,
    COALESCE(eo.`PorteCles`, 0) AS option_porte_cles_quantity,
    COALESCE(eo.`MagnetsSimple`, 0) AS option_magnets_simple_quantity,
    COALESCE(eo.livraison, 0) AS option_delivery,
    COALESCE(eo.duree, 0) AS option_duration_hours,
    (
          COALESCE(eo.`MurFloral`, 0)
        + COALESCE(eo.`Phonebooth`, 0)
        + COALESCE(eo.`LivreOr`, 0)
        + COALESCE(eo.`Fond360`, 0)
        + COALESCE(eo.`PanneauBienvenue`, 0)
        + COALESCE(eo.`PhotographeVoguebooth`, 0)
        + COALESCE(eo.`ImpressionVoguebooth`, 0)
        + COALESCE(eo.`DecorVoguebooth`, 0)
        + COALESCE(eo.`Holo3D`, 0)
        + COALESCE(eo.`PanneauFontaine`, 0)
        + COALESCE(eo.`VideoLivreOr`, 0)
        + CASE WHEN COALESCE(eo.magnets, 0) > 0 THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(eo.`PorteCles`, 0) > 0 THEN 1 ELSE 0 END
        + CASE WHEN COALESCE(eo.`MagnetsSimple`, 0) > 0 THEN 1 ELSE 0 END
    ) AS option_count,

    /* Prix connus lors de la proposition */
    e.prix_brut AS price_catalog_gross,
    COALESCE(e.reduc_product, 0) AS price_product_discount,
    COALESCE(e.reduc_all, 0) AS price_global_discount,
    (
          COALESCE(eo.`MurFloral_reduc_prix`, 0)
        + COALESCE(eo.`Phonebooth_reduc_prix`, 0)
        + COALESCE(eo.`LivreOr_reduc_prix`, 0)
        + COALESCE(eo.`Fond360_reduc_prix`, 0)
        + COALESCE(eo.`PanneauBienvenue_reduc_prix`, 0)
        + COALESCE(eo.`PhotographeVoguebooth_reduc_prix`, 0)
        + COALESCE(eo.`ImpressionVoguebooth_reduc_prix`, 0)
        + COALESCE(eo.`DecorVoguebooth_reduc_prix`, 0)
        + COALESCE(eo.`Holo3D_reduc_prix`, 0)
        + COALESCE(eo.`PanneauFontaine_reduc_prix`, 0)
        + COALESCE(eo.`VideoLivreOr_reduc_prix`, 0)
        + COALESCE(eo.magnets_reduc_prix, 0)
        + COALESCE(eo.`PorteCles_reduc_prix`, 0)
        + COALESCE(eo.`MagnetsSimple_reduc_prix`, 0)
    ) AS price_option_discount,
    (
          COALESCE(e.reduc_product, 0)
        + COALESCE(e.reduc_all, 0)
        + COALESCE(eo.`MurFloral_reduc_prix`, 0)
        + COALESCE(eo.`Phonebooth_reduc_prix`, 0)
        + COALESCE(eo.`LivreOr_reduc_prix`, 0)
        + COALESCE(eo.`Fond360_reduc_prix`, 0)
        + COALESCE(eo.`PanneauBienvenue_reduc_prix`, 0)
        + COALESCE(eo.`PhotographeVoguebooth_reduc_prix`, 0)
        + COALESCE(eo.`ImpressionVoguebooth_reduc_prix`, 0)
        + COALESCE(eo.`DecorVoguebooth_reduc_prix`, 0)
        + COALESCE(eo.`Holo3D_reduc_prix`, 0)
        + COALESCE(eo.`PanneauFontaine_reduc_prix`, 0)
        + COALESCE(eo.`VideoLivreOr_reduc_prix`, 0)
        + COALESCE(eo.magnets_reduc_prix, 0)
        + COALESCE(eo.`PorteCles_reduc_prix`, 0)
        + COALESCE(eo.`MagnetsSimple_reduc_prix`, 0)
    ) AS price_total_declared_discount,
    e.prix_proposed AS price_proposed,
    e.prix_proposed - e.prix_brut AS price_proposed_minus_gross,
    ROUND(e.prix_proposed / NULLIF(e.prix_brut, 0), 4) AS price_proposed_to_gross_ratio,

    /* Cible et resultat : ne jamais utiliser comme variables du modele initial */
    o.target_signed,
    o.outcome_class,
    o.target_training_eligible,
    o.target_price_training_eligible,
    e.status AS outcome_current_status,
    e.history_status AS outcome_status_history,
    e.signer_at AS signed_at_utc,
    TIMESTAMPDIFF(DAY, e.created_at, e.signer_at) AS signed_delay_days,
    e.prix_valided AS signed_price,
    e.prix_valided - e.prix_proposed AS signed_price_vs_proposed,

    /* Acompte/paiement : resultat financier, pas variable avant devis */
    ea.montant_acompte AS payment_deposit_amount,
    ea.mode_payement AS payment_method,
    ea.date_payement AS payment_date,
    ea.montant_restant AS payment_remaining_amount,

    /* Couts reels partiels disponibles apres prestation */
    epp.charge AS postpresta_recorded_charge,
    epp.membre_salary AS postpresta_member_salary,
    (
        COALESCE(epp.charge, 0) + COALESCE(epp.membre_salary, 0)
    ) AS postpresta_recorded_direct_cost_partial,
    CASE
        WHEN e.prix_valided IS NOT NULL THEN
            e.prix_valided
            - COALESCE(epp.charge, 0)
            - COALESCE(epp.membre_salary, 0)
        ELSE NULL
    END AS postpresta_margin_partial,
    epp.client_paid AS postpresta_client_paid,
    epp.members_paid AS postpresta_members_paid,

    /* Signaux disponibles seulement apres l'envoi */
    COALESCE(es.email_sent_count, 0) AS email_sent_count,
    COALESCE(es.email_open_count, 0) AS email_open_count,
    COALESCE(es.email_any_opened, 0) AS email_any_opened,
    es.email_first_sent_at AS email_first_sent_at_utc,
    es.email_first_opened_at AS email_first_opened_at_utc,
    TIMESTAMPDIFF(MINUTE, es.email_first_sent_at, es.email_first_opened_at)
        AS email_minutes_to_first_open,

    COALESCE(fs.followup_count, 0) AS followup_count,
    fs.followup_first_at AS followup_first_at_utc,
    fs.followup_last_at AS followup_last_at_utc,
    COALESCE(fs.followup_max_qualification, 0) AS followup_max_qualification,
    COALESCE(fs.followup_avg_qualification, 0) AS followup_avg_qualification,
    COALESCE(fs.followup_with_comment_count, 0) AS followup_with_comment_count

FROM app_event e
INNER JOIN app_client c
    ON c.id = e.client_id
INNER JOIN app_eventdetails ed
    ON ed.id = e.event_details_id
LEFT JOIN app_eventproduct ep
    ON ep.id = e.event_product_id
LEFT JOIN app_eventoption eo
    ON eo.id = e.event_option_id
LEFT JOIN app_eventacompte ea
    ON ea.id = e.event_acompte_id
LEFT JOIN app_eventpostprestation epp
    ON epp.id = e.event_post_presta_id
LEFT JOIN email_stats es
    ON es.event_id = e.id
LEFT JOIN followup_stats fs
    ON fs.event_id = e.id
LEFT JOIN client_history ch
    ON ch.event_id = e.id
INNER JOIN outcomes o
    ON o.event_id = e.id
ORDER BY e.created_at ASC;


/* ================================================================
   3. EXPORT DES COUTS GENERAUX

   Ces couts ne sont pas relies a un event dans le schema actuel.
   Ils doivent donc etre analyses separement et ne peuvent pas calculer
   automatiquement la marge exacte de chaque prestation.
   ================================================================ */

SELECT
    c.id AS cost_id,
    nc.name AS cost_name,
    c.type_cost,
    c.price_cost,
    c.created_at,
    c.frecency
FROM app_cost c
INNER JOIN app_namecost nc
    ON nc.id = c.name_cost_id
ORDER BY c.created_at ASC, c.id ASC;


/* ================================================================
   4. RESUME DE L'EXPORT ENTRAINABLE
   ================================================================ */

SELECT
    COUNT(*) AS total_requests,
    SUM(
        CASE
            WHEN e.signer_at IS NOT NULL
              OR e.prix_valided IS NOT NULL
              OR e.status IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media')
            THEN 1 ELSE 0
        END
    ) AS signed_requests,
    SUM(CASE WHEN e.status = 'Refused' THEN 1 ELSE 0 END) AS explicitly_refused_requests,
    SUM(
        CASE
            WHEN e.signer_at IS NULL
             AND e.prix_valided IS NULL
             AND e.status NOT IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media', 'Refused')
             AND ed.date_evenement < CURRENT_DATE()
            THEN 1 ELSE 0
        END
    ) AS expired_without_signature_requests,
    SUM(
        CASE
            WHEN e.signer_at IS NULL
             AND e.prix_valided IS NULL
             AND e.status NOT IN ('Acompte OK', 'Presta FINI', 'Post Presta', 'Sent Media', 'Refused')
             AND ed.date_evenement >= CURRENT_DATE()
            THEN 1 ELSE 0
        END
    ) AS still_open_requests,
    SUM(CASE WHEN e.prix_proposed IS NULL THEN 1 ELSE 0 END) AS missing_proposed_price,
    SUM(CASE WHEN c.how_find IS NULL OR TRIM(c.how_find) = '' THEN 1 ELSE 0 END)
        AS missing_request_source,
    SUM(CASE WHEN e.event_product_id IS NULL THEN 1 ELSE 0 END) AS missing_products,
    SUM(CASE WHEN e.event_option_id IS NULL THEN 1 ELSE 0 END) AS missing_options
FROM app_event e
INNER JOIN app_eventdetails ed
    ON ed.id = e.event_details_id
INNER JOIN app_client c
    ON c.id = e.client_id;
