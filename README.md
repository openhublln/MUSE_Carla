# CARLA Multi-Sensor Data Collection Pipeline

Ce pipeline vise à générer et visualiser des données synthétiques issues du simulateur CARLA. 
L’objectif est de collecter des données provenant de différents capteurs (caméra, lidar, radar) 
dans des scènes simulées et ensuite de pouvoir rejouer et analyser ces données grâce à des outils 
de visualisation interactifs. Cela permet aux chercheurs et développeurs de tester des algorithmes 
de perception et de fusion de capteurs dans un environnement contrôlé et répété.

## Requirements

- CARLA 0.10.0
- Python 3.7+
- Required Python packages:
  - numpy<2.0,>=1.24.4
  - open3d; python_version <= '3.11'
  - Pillow
  - matplotlib
  - pygame
  - PyYAML

## Architecture du pipeline

L’architecture globale du pipeline repose sur une séparation claire entre :
  - La configuration : centralisée dans "config.yml" pour une maintenance et une évolutivité simplifiées.
  - La collecte de données : gérée par "multi_sensor_collection.py", qui coordonne la simulation CARLA, 
    le spawn des acteurs, et la sauvegarde des données.
  - La visualisation : réalisée via "multi_sensor_replay.py" qui lit les données enregistrées et offre 
    une interface interactive.

## Fichier de configuration (config.yml)

Le fichier de configuration, "config.yml", centralise l’ensemble des paramètres de la simulation et 
la configuration des capteurs. Ce fichier YAML permet de spécifier :
  - Le nombre de scènes et la durée (nombre de ticks par scène).
  - Le chemin de sauvegarde pour les données collectées.
  - La liste des capteurs avec leur type, blueprint, attributs et position/rotation (transformation) 
    dans la scène.
Grâce à cette approche centralisée, il est très facile d’ajouter, modifier ou supprimer des capteurs 
sans changer le code des scripts de collecte ou de replay.

## Script de collecte multi-capteurs (multi_sensor_collection.py)

Ce script est chargé de lancer la simulation dans CARLA et de collecter les données des différents capteurs 
définis dans "config.yml". Il effectue les tâches suivantes :
  - Connexion au serveur CARLA et configuration du mode synchrone.
  - Création d’un dossier par scène et par capteur à partir du fichier de configuration.
  - Spawn du véhicule et attachement des capteurs (caméra, lidar, radar).
  - Lancement de la simulation pour un nombre de ticks déterminé, tout en s’assurant que chaque tick 
    reçoit les données de tous les capteurs.
  - Enregistrement des données sous forme d’images (pour les caméras) et de fichiers numpy (.npy) 
    pour les mesures lidar et radar.
Ce script permet une collecte modulaire et flexible des données, facilitée par la configuration centralisée.

## Script de replay multi-capteurs (multi_sensor_replay.py)

Ce script sert à rejouer et visualiser les données enregistrées durant la collecte. Ses principales fonctions
sont :
  - Chargement de la configuration "config.yml" pour déterminer quels capteurs afficher. Par défaut, les 
    données collectées dans la scene_1 vont être visualisées. Si une autre scène veut être visualisée,
    il faut le préciser via l'argument de la ligne de commande.
  - Recherche des données communes (timestamps) parmi les différents capteurs afin de synchroniser 
    l’affichage.
  - Affichage des données dans une fenêtre Pygame, avec un arrangement sous forme de grille. Une 
    fonction utilitaire (scale_to_fit) permet de redimensionner les images de façon à préserver leur 
    aspect ratio et à centrer chaque modalité dans sa zone d’affichage.
  - Navigation interactive entre les frames avec prise en charge du mode autoplay et des commandes 
    clavier.
L’approche assure que quels que soient le nombre et le type de capteurs spécifiés, leur affichage est 
proportionnel et non déformé.