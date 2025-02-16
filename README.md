# Implémentation et validation de réseaux bayésiens dynamiques d’ordre T avec pyAgrum

## Description

Les réseaux bayésiens dynamiques (dBN) sont une extension des réseaux bayésiens qui mo-
délisent des processus stochastiques. Ils représentent les dépendances probabilistes entre les
variables aléatoires d’un système à travers des tranches temporelles, en tenant compte des
relations probabilistes (ou causales) entre ces variables. Ces modèles sont particulièrement
adaptés à des applications dans des domaines tels que la modélisation de séries temporelles,
la prédiction de systèmes stochastiques ou encore l’inférence sur des processus biologiques,
financiers ou industriels.

Une propriété clé des dBN est la propriété de Markov, qui stipule que l’état futur du système
dépend uniquement d’un nombre limité d’états passés. Dans le cadre de ce projet, cette
propriété sera représenté par l’ordre T , permettant d’incorporer des dépendances sur T
tranches temporelles précédentes. Cette flexibilité ouvre des possibilités intéressantes pour
modéliser des phénomènes complexes, mais pose des défis particuliers autant en inférence qu’en
apprentissage.

Ce travail comprend plusieurs aspects :

- Développement d’algorithmes d’inférence conditionnelle pour les dBN d’ordre T.
- Exploration de méthodes d’apprentissage supervisé et semi-supervisé sur des données temporelles.
- Expérimentations sur des jeux de données synthétiques et réels.
- Visualisation et documentation des modèles obtenus.

## Équipe
Ce projet est réalisé par Jad CHAMSEDDINE et Jesus Alejandro GOMEZ URZUA, étudiants du cours LU2IN013, 
sous la supervision de Pierre-Henri WUILLEMIN.
