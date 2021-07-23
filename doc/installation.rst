Instalation de PyTools
======================

Installation de l'environnement virtuel
---------------------------------------

La documentation ci présente est générée avec la version 3.0.3 de sphinx qui est installé depuis PyPI
Pour installer un environnement virtuel qui contient tous les paquets du fichier requirements.txt :

.. code-block:: bash
        
   cd .../PyTools3/
   python3 -m venv venv
   
   # pour activer l'environnement lancer :
   source venv/bin/activate

Dans ce terminal toutes les consoles pythons et modules python s'exécuteront avec l'environnement venv.
Il est temps d'importer les paquets python nécessaires dans l'environnement :

.. code-block:: bash

   pip install --upgrade pip
   pip install -r requirements.txt
   
(les paquets seront installés dans l'environnement activé, donc venv si tout va bien)
Attention : il est nécessaire d'avoir installé tous ces paquets python pour que sphinx puisse s'exécuter correctement, il est
donc fortement recommandé de passer par l'installation d'un environnement virtuel présentée à l'instant.

Pour désactiver un environnement virtuel, lancer la commande :

.. code-block:: bash

   deactivate

Génération de la documentation avec Sphinx
------------------------------------------

Sphinx a été téléchargé avec les paquets python du fichier requirements.

Pour générer la documentation en html lancer : 

.. code-block:: bash

   make html

Pour la générer en latex (cassé pour le moment) lancer :  

.. code-block:: bash

   make latexpdf

Pour installer les extensions jupyter sur la machine, lancer :

Utiliation de Jupyter
---------------------

Jupyter est aussi téléchargé en tant que paquet python, donc il doit être dans ``venv``.

La première étape est d'ajouter ``venv`` en tant que kernel pour jupyter :

.. code-block:: bash

   source venv/bin/activate
   ipython kernel install --name "venv" --user

Cepandant pour installer un gestionnaire d'extensions (bien pratique), il faut lancer les commandes suivantes :

.. code-block:: bash

   source venv/bin/activate
   jupyter contrib nbextension install --user
   jupyter nbextensions_configurator enable --user

Il faut activer au moins deux plugin pratique : table of content (2) via le plugin manager qui se trouve dans les onglets à l'ouverture d'une session jupyter, et hide_code, qui permet au choix de cacher le code, le resultat ou le numéro de cellule lors de l'export du Notebook. Ce plugin propose aussi un export au format slides. Pour l'installer lancer les commandes suivantes :

.. code-block:: bash

   source venv/bin/activate
   pip install hide_code
   jupyter nbextension install --py hide_code
   jupyter nbextension enable --py hide_code

Pour lancer jupyter procéder de la manière suivante :

.. code-block:: bash

   cd .../PyTools3
   source venv/bin/activate
   jupyter notebook

Pour lancer un test de non régression sur des notebooks de test :

.. code-block:: bash

   cd mon/chemin/vers/PyTools3/fiche_test/
   pytest --nbval

Pour avoir un rapport de coverage des tests du code :

.. code-block:: bash

   cd mon/chemin/vers/PyTools3/fiche_test
   pytest --nbval --cov=../commons/ . --cov-report=html

Pour lancer les test, le rapport de coverage, la documentation et inclure le rapport de test à la doc,
il faut lancer le script suivant :

.. code-block:: bash

   cd mon/chemin/vers/PyTools3/
   ./test_and_doc.sh

