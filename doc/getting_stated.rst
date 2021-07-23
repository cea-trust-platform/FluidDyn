Getting started
===============

Introduction
------------

Cette bibliothèque est destinée à réaliser des opérations de post-traitement en python des fichiers de calcul de IJK.
Elle implémente des objets de plusieurs niveaux et des fonctions variées :

* un champ de base :py:class:`~commons.Tools.Field` qui hérite de ``numpy.ndarray``, et implémente des attributs supplémentaire ainsi que des méthodes gérants ces attributs. On peut aussi réaliser tout un tas d'opération classique ``numpy`` dessus. Cette objet a été construit dans un but de recherche robuste et reproductible. L'un des attributs est ``tex``. Cet attribut a pour but d'enregsitrer les étapes qui ont menées à la construction de cet objet. Pour plus de détails cf. la partie `Fiches de test`.
* un objet :py:class:`~commons.DNSTools.Simu` qui est une classe vouée à réunir les champs résultant d'une simulation. Elle possède des méthodes destinées à charger les champs voulus.


Présentation du code et du workflow de post-traitement
------------------------------------------------------

Chargement des fichiers
~~~~~~~~~~~~~~~~~~~~~~~

* Pour les paramètres de simulation : en utilisant :class:`~DNSTools.Simu` de ``DNSTools``, on précise le numéro du fichier thermique qui correspond
* Pour les statistiques : transformation des fichiers statistiques_*.txt  et statistiques_thermiques_x_*.dt\_ev en fichier dt\_ev avec ``BuildStats``
* Pour les champs scalaires, vectoriels, tensoriels, etc : utilisation de la méthode :meth:`~commons.Tools.Field.initFromFile` de la classe :class:`~commons.Tools.Field` de ``Tools``
* Bientôt encapsulation du chargement dans la classe :class:`~DNSTools.Simu` de ``DNSTools``

Utilisation d'un Field
~~~~~~~~~~~~~~~~~~~~~~

* Opération mathématiques classiques
* Opérations tensorielles
* Suivi des opérations
* Aperçu des attributs
* Plot rapide et modulaire avec légende adaptable

Chargement des Field dans Simu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* chargment des statistiques avec Simu
* chargement d'un med avec Simu

Plot des Fields / Simu
~~~~~~~~~~~~~~~~~~~~~~

* les Field possèdent une méthode :py:meth:`~commons.Tools.Fiel.plot` adaptative
* pour plotter plusieurs Fields ensemble, utiliser les fonctions du module :py:mod:`commons.PlotField`
