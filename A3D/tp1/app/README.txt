Rendu A3D Louis Forestier

Le code a été testé sur émulateur avec version d'API 28 et 30, ainsi que sur mobile avec Android version 9 (API 28).

Contrôles :
Pour se déplacer dans l'espace, j'ai implanté des contrôles simulant des joysticks.
Pour se faire, j'ai enregistré la position du (des) doigt(s) lorsqu'on le(s) pose sur l'écran.
Ensuite, lors d'un mouvement, je récupère la nouvelle position, je calcule les delta entre la position appuie et la nouvelle et les applique à chaque frame,
grâce à la fonction step de Scene.
Sur la partie gauche de l'écran, on contrôle le déplacement de la caméra, la translation.
Sur la partie droite de l'écran, on contrôle l'axe de la caméra, la rotation. La rotation est bornée pour éviter de retourner l'angle de vue.
Bien entendu, il est possible et même conseillé d'utiliser les deux doigts simultanément pour se déplacer et tourner en même temps.
Par conséquent, ces contrôles sont moins efficaces sur l'émulateur.

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

La classe Mesh stocke les tableaux de sommets et de triangles.
Elle encapsule la création des VBOs dans la méthode initGraphics et l'affiche dans la méthode draw.
Je stocke mes indices de sommets dans le tableau des triangles sur des int.
En effet, je me suis aperçu que pour la plupart des modèles 3D de Stanford (comme l'Armadillo) les shorts ne suffisaient pas car je dépassais 32767 sommets, la valeur maximale d'un short.

La classe Transform est composé de 3 Vec3f représentant chacun la position, la rotation et l'échelle d'un objet.
La transforme est toujours initialisée avec ses positions et rotations à 0  et l'échelle à 1.
Ensuite, les modifications sont faites avec des setters sur chaque composante des Vec3f.
Ils sont faits comme les setters d'un patron de conception Builder pour retourner l'objet après l'avoir modifié, afin de pouvoir enchaîner les appels aux setters.
Je dois avouer avoir toujours un doute sur l'ordre des transformations notamment sur l'ordre des rotations.

Tous les maillages que j'ai réalisé hérite donc de la classe Mesh.
Les différentes formes réalisés sont :
    - la sphère (par longitude et latitude et par subdivision), classe Sphere
    - le cube, classe Cube
    - le cylindre "plein", classe Cylinder
    - le tore, classe Donut
    - le frustum ou tronc, classe Frustum
    - le cylindre "vide" avec faces internes et externes, classe Pipe
    - le plan, rectangle de taille 10x10, classe Plane
    - la pyramide, classe Pyramid
    - la capsule, classe Tictac

Le code de la plupart des formes est très largement inspiré et adapté depuis le code de la sphère par longitude*latitude.
Pour la sphère par subdivision, j'ai utilisé une map avec des paires d'entiers comme clés pour éviter la duplication des points.
Sachant que je trie pour toujours crée ma clé avec le plus petit entier en premier, j'ai toujours mes points dans le même ordre lorsque je crée ma clé.

Comme demandé dans le TP3, j'ai aussi une classe Ball qui a une Sphere static.
Je redéfinis la méthode initGraphics pour qu'elle fonctionne un peu comme un singleton.
J'utilise un booléen pour savoir si la sphère a été initialisée. Si ce n'est pas le cas, alors je l'initialise.
J'ai défini une méthode onPause pour mettre le booléen à false, que j'appelle dans la méthode onPause de MainActivity.
En effet, j'ai pu constater dans la documentation que le contexte OpenGL n'était pas conservé par défaut lors du onpause de GLSurfaceView.
J'ai aussi mis en commentaire la ligne qui permet de le conserver malgré tout, mais j'ai jugé cette méthode un peu trop gourmande.

Au niveau des pièces, j'ai un objet Room qui me permet de faire des pièces avec des portes dépendantes des paramètres du constructeur.
Ces pièces sont composés de GameObject avec un Mesh Plane pour le plafond et le sol et de GameObject Wall pour les murs.
Un Wall est composé de plusieurs Planes. Il a aussi un outline qui est un Square. C'est la seul manière que j'ai trouvé sans trop casser ma conception et l'encapsulation
pour réafficher les contours des pièces.
J'ai gardé des attributs de longueur, largeur et hauteur pour les pièces car je veux pouvoir conserver la taille des portes, je ne peux donc pas appliquer de mise à l'échelle.
J'ai conservé les Room comme définies dans le tp dans la classe OldRoom.

J'ai aussi une classe OBJImporter avec une méthode static pour importer un fichier OBJ.
Pour l'instant, l'importation est assez naive car elle ne supporte qu'un objet par OBJ et ne récupère que les sommets et leur organisation pour créer les faces.

Scène :
La scène est composée de 4 pièces disposées de cette manière :

    N                                           *********** ***********
   O+E                                          *                     *
    S                                           *                     *
                                                *                     *
                                                *          4          *
                                                *                     *
                                                *                     *
                                                *                     *
                          ********************************* ***********
                          *                     *                     *
                          *                     *                     *
                          *                     *                     *
                          *          1                     3          *
                          *                     *                     *
                          *                     *                     *
                          *                     *                     *
      ******************************* *********************************
      *                                                           *
      *                                                           *
      *                                                           *
      *                              2                            *
      *                                                           *
      *                                                           *
      *                                                           *
      *************************************************************

Le point de vue de départ est en 0,0,0 au milieu de la pièce 1, qui regarde vers le nord (vers les négatifs de l'axe Z).
Dans la première pièce, il y a 2 objets Ball.
Dans la deuxième pièce, il y a une capsule, un cube, un tore, un cylindre plein, un cylindre vide et une pyramide (avec suffisament de quartiers pour que ça ressemble à un cône).
Dans la troisième pièce, il y a l'Armadillo.
Dans la quatrième pièce, il y a un frustum.