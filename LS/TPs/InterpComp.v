Require Import PeanoNat.
Import Nat.

(** **** Exercice : (lang_interp_complex) *)

(** L'objectif de cet exercice est de modéliser un mini langage impératif sur
    les expressions arithmétiques des nombres complexes entiers. Celui-ci est très
    simple en particulier il n'y pas de variables.

    Les nombres complexes sont représentés par un couple d'entiers.

    L'objectif est de lui donner une sémantique à l'aide d'un
    interpréteur. Celui-ci va modifier l'état de la mémoire (représenté
    simplement par un nombre complexe) en fonction des commandes du programme afin de
    calculer un état qui sera la valeur calculée par le programme.

    Ce langage est défini ci-dessous par le type inductif ProgComp. La
    signification informelle de chaque instruction est indiquée.

    On remarquera que les deux composantes du nombre complexe sont données
    séparément comme arguments du constructeur (i. e il n'y a pas de virgule
    entre les deux composantes).

*)
Inductive ProgComp :=
  | Done                                   (* Ne modifie pas l'état. *)
  | AddThen (a b : nat) (p : ProgComp)     (* Ajoute le complexe [(a b)] à l'etat et ensuite exécute [p]. *)
  | MulThen (a b : nat) (p : ProgComp)     (* Multiplie l'état par le complexe [(a b)] et ensuite exécute [p]. *)
  | SetToThen (a b : nat) (p : ProgComp)   (* Met l'état a [(a b)] et ensuite exécute [p]. *)
.

(** Expliquez ce que fait le programme 

    MulThen 3 4 (AddThen 5 7 Done)


    à quoi sert l'instruction Done ?
 *)

(*
    Ce programme multiplie l'état par le complexe (3 4) (qui est vide) puis ajoute le complexe (5 7) à l'état et se termine.
    Done est l'équivalent du skip du programme impératif IMP vu en cours, elle permet d'avoir un cas d'arrêt sur notre type récursif.
 *)

(** Expliquez ce que fait le programme

    SetToThen 5 9 (MulThen 3 2 (AddThen 7 11 Done))

    à quoi sert l'instruction SetToThen ?
*)

(*
    On initialise l'état de la mémoire à (5 9), on le multiplie par (3 2), puis on ajoute (7 11) et on termine le programme.
    SetToThen permet d'initialiser l'état de la mémoire.
*)

(** À l'aide de ce langage, écrivez : 
    1. un programme qui additionne les nombres complexes (2, 1) et (4, 8).
    2. un programme qui multiplie les nombres complexes (2, 5) et (3, 10) puis
    qui ajoute (5, 1).

    Faites bien attention à vous assurer que votre programme calcule bien ce qui
    est attendu quel que soit l'état initial de la mémoire.
 *)

(*
    1. SetToThen 2 1 (AddThen 4 8 Done)
    2. SetToThen 2 5 (MulThen 3 10 (AddThen 5 1 Done))
*)

(** Définissez une fonction (i.e. un interpréteur) [run] qui prend en arguments
    un programme [p] et un état initial [(a b)] (i.e. un nombre complexe entier) de la mémoire et
    qui renvoie l'état (i.e. le nombre complexe entier) obtenu après l'exécution du programme
    [p]. L'exécution des instructions de programme se traduit par l'exécution
    des opérations correspondantes sur l'entier qui représente l'état de la
    mémoire.

    Pour rappel l'addition de deux nombres complexes (a b) et (c d) est ((a+c) (b+d)) 
    et la multiplication est ((a*c - b*d) (b*c + a*d))   
 *)

Fixpoint run (p : ProgComp) (a b : nat) : nat*nat :=
    match p with
    | Done => (a,b) 
    | AddThen c d p' => run p' (a+c) (b+d)
    | MulThen c d p' => run p' (a*c - b*d) (b*c + a*d)
    | SetToThen c d p' => run p' c d
    end.

(** On peut maintenant écrire des tests unitaires. Votre interpréteur doit
    vérifier les exemples suivants dont vous prouverez la satisfaction : 
 *)

Example run_Example1 : run Done 1 0 = (1,0).
Proof. simpl. reflexivity. Qed.


Example run_Example2 : run (MulThen 5 2 (AddThen 2 7 Done)) 1 0 = (7, 9).
Proof. simpl. reflexivity. Qed.

Example run_Example3 : run (SetToThen 3 6 (MulThen 2 4 Done)) 10 3 = (0, 24).
Proof. simpl. reflexivity. Qed.

(** Vous pouvez maintenant utiliser votre interpréteur pour exécuter un
    programme. En Coq cela se fait à l'aide de la commande Compute qui exécute
    le terme qui lui passé en argument dans une machine virtuelle. 

    Expliquez à quoi est due la différence de résultat lors de l'exécution des
    deux programmes ci-dessous. Donnez les modifications nécessaires pour
    s'assurer d'obtenir le même résultat.  
*)
Compute (run (MulThen 5 3 (AddThen 2 8 Done)) 4 5).

Compute (run (MulThen 5 3 (AddThen 2 8 Done)) 3 9).

(*
    Les paramètres d'initialisation de l'état de la mémoire sont différents entre les 2 programmes.
    Le premier initialise à 4 5 donc on obtient (7,45) à la fin des calculs.
    Le deuxième initialise à 3 9 donc on obtient (2,62) à  la fin des calculs.
    Si on veut obtenir le même résultat alors il faut partir du même état de la mémoire.
    Pour cela, soit on met les mêmes paramètres à run, soit on commence le programme par un SetToThen avec les mêmes paramètres.
*)

