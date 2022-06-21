Rendu A3D Louis Forestier

Structure de l'archive :
```
app -
    |- src -
           |- main -
                   |- assets (contient les fichiers glsl)
                   |- java -
                           |- fr.univ_poitiers.dptinfo.algo3d -
                                                              |- gameobject (package contenant les classes de GameObject, ainsi que les classes Component et Transform)
                                                              |- mesh (package contenant tout ce qui est lié aux maillages)
                                                              |- objimporter (package contenant les classes liés  à l'importation d'obj)
                                                              |- shaders (package contenant les classes de Shaders et ce qui est lié à l'éclairage)
                                                              |- MainActivity.java
                                                              |- MyGLRenderer.java
                                                              |- MyGLSurfaceView.java
                                                              |- Scene.java
                                                              |- Vec3f.java
                   |- res (contient les ressources, notamment les textures et les .obj)
    |- build.gradle
    |- README.md
```
Les classes sont commentées avec des commentaires types Javadoc.

Le code a été testé sur émulateur avec version d'API 30, ainsi que sur mobile avec Android version 9 (API 28).
Avec API inférieur à 30, j'ai plusieurs erreurs que je n'ai pas comprises :
- avec émulateur API 25 : 
        "a vertex attribute array is uninitialized. Skipping corresponding vertex attribute" lors de l'appel de renderShadow dès que j'appelle la méthode glDrawElements pour n'importe quel objet. Pourtant, je n'ai pas cette erreur si je commente l'appel à renderShadow donc tout se passe correctement lors de renderScene.
- avec émulateur API 28 : 
        "a vertex attribute index out of boundary is detected. Skipping corresponding vertex attribute. buf=0xe79a5a10
        Out of bounds vertex attribute info: clientArray? 0 attribute 2 vbo 216 allocedBufferSize 1648 bufferDataSpecified? 1 wantedStart 0 wantedEnd 20416"
        Encore déclenché par renderShadow


Contrôles :
Pour se déplacer dans l'espace, j'ai implanté des contrôles simulant des joysticks.
Pour se faire, j'ai enregistré la position du (des) doigt(s) lorsqu'on le(s) pose sur l'écran.
Ensuite, lors d'un mouvement, je récupère la nouvelle position, je calcule les delta entre la position appuie et la nouvelle et les applique à chaque frame,
grâce à la fonction step de Scene.
Sur la partie gauche de l'écran, on contrôle le déplacement de la caméra, la translation.
Sur la partie droite de l'écran, on contrôle l'axe de la caméra, la rotation. La rotation est bornée pour éviter de retourner l'angle de vue.
Bien entendu, il est possible et même conseillé d'utiliser les deux doigts simultanément pour se déplacer et tourner en même temps.
Par conséquent, ces contrôles sont moins efficaces sur l'émulateur.
Pour réinitialiser la position de l'utilisateur, il suffit de toucher l'écran avec plus de 2 doigts.
Cela réinitialise aussi les joysticks à la position actuelle des 2 premiers doigts.

Je me suis grandement inspiré de l'architecture de Unity.

Ainsi, les objets que j'affiche sont de la classe GameObject.
Cette classe a plusieurs attributs :
- une transform "transform" de la classe Transform
- une liste d'objets enfants "children"
- une liste de composants de la classe Component
- un objet parent "parent"

Les composants réalisés sont :
- MeshFilter qui est un conteneur de maillage
- MeshRenderer qui affiche le maillage du MeshFilter en fonction des attributs du material
- Light qui permet d'émettre de la lumière

Au niveau des shaders, j'ai laissé toutes celles que j'ai créé.
La plus abouti est la classes ShadowShaders avec les glsl shadow_frag et shadow_vert.
J'y ai fait un début d'implantation des ombres. 
En effet, la lumière directionnelle de la scène projette des nombres dans une certaine zone.
Pour l'instant cette zone est fixe, centrée sur la position de cette lumière.
Les autres lumières ne projettent pas d'ombre.

Scène :
La scène est composée de 4 pièces disposées de cette manière :
```
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
```
Le point de vue de départ est en 0,0,0 au milieu de la pièce 1, qui regarde vers le nord (vers les négatifs de l'axe Z).
Dans la première pièce, il y a 2 objets Ball.
Dans la deuxième pièce, il y a une capsule, un cube, un tore, un cylindre plein, un cylindre vide et une pyramide (avec suffisament de quartiers pour que ça ressemble à un cône).
Dans la troisième pièce, il y a 2 Armadillos, 1 avec les normals calculées pour du Smooth Shading et l'autre avec les normals fournies dans l'obj, et un XYZDragon avec normals calculées pour du Flat Shading.
Dans la quatrième pièce, il y a un frustum.
