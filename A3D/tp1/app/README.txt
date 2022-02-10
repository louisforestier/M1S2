Rendu A3D Louis Forestier

Le code a été testé sur émulateur avec version d'API 30 et 28, ainsi que sur mobile avec Android version 9 (API 28).

J'ai fini le TP3 et fait quelques ajouts de forme et une modification de la conception globale de ma gestion des objets.

Je me suis grandement inspiré de l'architecture de Unity.

Ainsi, les objets que j'affiche sont de la classe GameObject.
Cette classe a plusieurs attributs :
    - un maillage "mesh" de la classe Mesh
    - une transform "transform" de la classe Transform
    - un tableau de float "color" représentant sa couleur
    - une liste d'objets enfants "children"
    - un objet parent "parent"
J'ai utilisé cette construction pour séparer le calcul des points et des triangles, des transformations de la ModelMatrix.
L'attribut children me permet de réaliser des objets composés de plusieurs sous-objets.
L'attribut parent n'est pour l'instant pas utilisé mais existe en prévision, car je pense qu'il peut avoir un intérêt.

Les méthodes présentes permettent de définir un maillage ou d'accéder à la transforme pour la modifier, etc.
La méthode initGraphics permet d'initialiser les VBOs du maillage de l'objet et des sous-objets, s'il y en a.
La méthode draw permet de définir une ModelViewMatrix pour l'objet courant à partir de la ModelMatrix de la transform
 et de la ViewMatrix et d'afficher le maillage de l'objet et d'afficher les sous-objets en leur transmettant la MVM calculée.

La classe Mesh stocke les tableaux de sommets et de triangles
Elle encapsule la création des VBOs dans la méthode initGraphics et l'affiche dans la méthode draw.

La classe Transform est composé de 3 Vec3f représentant chacun la position, la rotation et l'échelle d'un objet.
La transforme est toujours initialisée avec ses positions et rotations à 0  et l'échelle à 1.
Ensuite, les modifications sont faites avec des setters sur chaque composante des Vec3f.
Ils sont faits comme les setters d'un patron de conception Builder pour retourner l'objet après l'avoir modifié, afin de pouvoir enchaîner les appels aux setters.

Tous les maillages que j'ai réalisé hérite donc de la classe Mesh.
Les différentes formes réalisés sont :
    - la sphère (par longitude et latitude et par subdivision)