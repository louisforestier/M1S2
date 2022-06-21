# Projet Lancer de rayon

Autheur : Louis Forestier

Le code a été réalisé avec Java 8 et utilise le framework JavaFX inclus dans cette version de Java.\
En utilisant cette version, il n'y a pas besoin de spécifier le module JavaFX, on peut compiler et exécuter comme une programme Java standard.

Deux classes possèdent une méthode main, les classes JavaTGA et Main.\
La classe JavaTGA produit l'image au format tga et la classe Main fait de même mais affiche aussi l'image dans une fenêtre JavaFX.

L'application suit le principe de Modèle Vue Contrôleur.

J'ai parallélisé le lancer de rayon par pixel de plusieurs manières en utilisant les streams Java.
Sur ma machine, la méthode la plus performante est renderParallelNestedLoops  qui me permet d'avoir une accélération entre 3 et 4 par rapport à l'algorithme séquentiel.
L'appel de la méthode de rendu de la scène est entouré par des méthodes pour chronométrer le temps d'exécution.

J'ai pris le parti de mélanger les couleurs obtenues par la transmission, la réflexion et le matériau.
Ce n'est pas parfait, j'ai vu qu'il existait le coefficient de Fresnel pour mélanger réflexion et transmission mais je n'ai pas eu le temps de l'implanter.
Le principal problème est qu'il n'est plus possible d'obtenir des objets qui réfléchissent ou qui transmettent à 100% la lumière, ils ont forcément une couleur.
Une solution pourrait être de ne pas mettre de couleur pour ces objets.
