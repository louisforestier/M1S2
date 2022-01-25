Require Import NaturalDeduction.
Require Import Hoare.
Require Import Lemmas. (* Quelques propriétés utiles dans les exercices *)
(* Vous devez récupérer et compiler le fichier NaturalDeduction.v, Hoare.v et Lemmas.v *)

(* De quoi pouvoir travailler avec les entiers *)
Require Import PeanoNat.
Import Nat.

(* La théorie décidable de l'arithmétique linéaire, très utiles pour nos 
   exemples *)
Require Import Lia.

(** * La logique de Hoare *)

(** L'objectif est de vérifier des propriétés de fragments de code d'un langage
   impératif.  *)

(** * Le langage IMP *)

(**
   IMP est un mini langage impératif dont la grammaire est :

   Nombres                n : Nat
   Variables              x : String
   Expressions            e ::= n | x | e + e | e - e | e * e
   Expressions booléennes b ::= true | false | (!b) | (b && b) | (b || b) | (e <? e) | (e = e)
   Commandes              c ::= Skip | x ::= e | c;;c | If b Then c Else c Fi | While b Do c Od
 *)

(** **** Exercice : (IMP_pgm) *)
(**  Écrivez un programme en IMP qui  :

     1. Ajoute 1 à une variable
     2. Échange les contenus de deux variables
     3. Calcule la valeur absolue d'une valeur contenue dans une variable

 *)

(* 
1.
  x ::= x + 1
2.
  z ::= y;;
  y ::= x;;
  x ::= z

3.
  If  x >= 0 
  Then x
  Else -x


*)
(** [] *)


(** * Les triplets de Hoare *)

(** Un triplet de Hoare est de la forme

      {{phi}}P{{psi}}

   où P est un programme et phi et psi sont des formule de la logique du premier
   ordre avec l'ensemble F des symboles de fonctions , d'arité 1 (unaire) -,
   d'arité 2 (binaire) +, -, *, / et l'ensemble P des prédicats < et = (on
   s'autorise des abréviations comme x < 4 \/ x = 4 qui s'abrège en x <= 4).

   Les formules phi et psi portent sur des états. Un état est une représentation
   de la mémoire. Une manière simple de voir les choses est de considérer un
   état comme une liste de couples formé d'un nom de variable et de la valeur
   associée à cette variable. Différentes représentation des états de la mémoire
   sont possibles, mais il s'agit toujours d'un modèle de F et P (On n'a pas vu
   la notion de modèle pour un ensemble de symboles de fonctions et de
   prédicats. Cette notion permet de définir la sémantique de la logique du
   premier ordre, notion que l'on n'a pas abordée).

   Les formules phi et psi sont la spécification du programme P. Par exemple, la
   spécification d'un programme P qui calcule un nombre dont le carré est plus
   petit qu'un autre nombre x est :

         {{x > 0}}P{{y*y < x}}.

   La signification d'un triplet {{phi}}P{{psi}} est :

     Si le programme P s'exécute à partir d'un état qui satisfait phi et si P
     termine alors l'état d'arrivée satisfait psi.

  Dans ce cas on dit que le triplet satisfait le relation de correction
  partielle que l'on note |= et on écrit |= {{phi}}P{{psi}}.

  On dit aussi que le programme P satisfait sa spécification.

  Ainsi notre exemple signifie que si l'on exécute P à partir d'un état tel que
  x > 0, et si P termine, alors l'état obtenu satisfait y * y < x.

  La condition que P termine est importante car un programme qui ne termine pas
  satisfait n'importe quelle spécification. Ce qui n'est pas très utile. 

  Il est possible de développer la logique de Hoare pour une relation de
  correction totale dans ce cas on impose que le programme P termine à coup
  sûr. Cependant, on commence souvent par prouver la correction partielle d'un
  programme puis sa terminaison et aussi les règles de déduction sont un peu
  différentes si l'on veut obtenir que le programme satisfait la relation de
  correction totale.
      
*)

(** * Les règles de déduction *)

(**
    Revoir les règles de déduction du cours. 
 *)

(** Les tactiques qui correspondent aux règles de déduction sont :

       * Pour l'affectation : Hoare_assignment_rule

       * Pour la conséquence : Hoare_consequence_rule_left et
         Hoare_consequence_rule_right qui s'utilisent en donnant en argument une
         formule, par exemple [Hoare_consequence_rule_left with (y + 1 < 4)]

       * Pour la séquence :  Hoare_sequence_rule qui s'utilise en donnant en
         argument une formule, par exemple [Hoare_sequence_rule with (z = x +
         y)]

       * Pour la conditionnelle : Hoare_if_rule

       * Pour la boucle : Hoare_while_rule

 *)

(**
   Pour trouver la suite de règles qu'il faut utiliser pour prouver la validité
   d'un triplet, il est nécessaire de décorer le programme.

     Voir exemples du cours

*)

(** * La règle de l'affectation *)

(** **** Exercice : (assign) *)
(** Écrivez la substitution à réaliser dans la précondition et prouvez la
   validité des triplets suivants : *)

Lemma ex11 (x : nat) :
  {{x + 1 < 1}}
     x ::= x + 1
  {{x < 1}}.
Proof.
  Hoare_assignment_rule.
Qed.


Lemma ex12( x y : nat) :
  {{x * x > 1}}
     y ::= x * x 
  {{y > 1}}.
Proof.
  Hoare_assignment_rule.
Qed.


(** [] *)

(** * La règle de la conséquence *)

(** **** Exercice : (weakening) *)
(** Trouvez une précondition plus faible qui permet d'appliquer la règle de
  l'affectation et prouvez la validité des triplets suivants : *)

(*
  {{True}}
  {{5 = 5}}
    x ::= 5
  {{x = 5}}.
*)
(*
Search add_comm.
Print add_comm.
About add_comm.*)
(*Dans coqide queries, emacs find*)

Lemma ex21 (x : nat) :
  {{True}}
    x ::= 5
  {{x = 5}}.
Proof.
  Hoare_consequence_rule_left with (5 = 5).
  Hoare_assignment_rule.
Qed.

(*
  {{y = 0 }}
  {{y + 1 = 1}}
    x ::= y + 1
  {{x = 1}}.

*)
Lemma ex22 (x y : nat) :
  {{y = 0 }}
    x ::= y + 1
  {{x = 1}}.
Proof.
  Hoare_consequence_rule_left with (y + 1 = 1).
  Impl_Intro.
  rewrite H.
  reflexivity.
  Hoare_assignment_rule.
Qed.

(*
{{y < 3}}
{{y+1 < 4}}
  y ::= y + 1
{{y < 4}}.
*)
Lemma ex23 (y : nat) :
  {{y < 3}}
    y ::= y + 1
  {{y < 4}}.
Proof.
  Hoare_consequence_rule_left with (y+1 < 4).
  Impl_Intro.
  lia.
  Hoare_assignment_rule.
Qed. 
  
Lemma ex24 (x y : nat) :
  {{x > 0}}
    y ::= x + 1
  {{y > 1}}.
Proof.
  Hoare_consequence_rule_left with (x+1 > 1).
  Impl_Intro. 
  lia.
  Hoare_assignment_rule.
Qed.  


(** [] *)


(** * La règle de la séquence *)

(** **** Exercice : (sequencing) *)
(** Pour chaque triplet, décorez le programme et prouvez le triplet associé en utilisant la
   règle de la séquence.  *)

(*
Votre programme décoré

{{ n = p /\ m = q }}
  h ::= n;;
  {{h = p /\ m = q}}
  n ::= m;;
  {{h = p /\ n = q}}
  m ::= h
{{ n = q /\ m = p }}.

*)

Lemma ex31 (n m p q h : nat) :
{{ n = p /\ m = q }}
  h ::= n;;
  n ::= m;;
  m ::= h
{{ n = q /\ m = p }}.
Proof.
  Hoare_sequence_rule with (h = p /\ n = q).
  Hoare_sequence_rule with (h = p /\ m = q).
  Hoare_assignment_rule.
  Hoare_assignment_rule.
  Hoare_consequence_rule_left with ( n = q /\ h = p ).
  Impl_Intro.
  And_Intro.
  And_Elim_2 in H.
  exact H0.
  And_Elim_1 in H.
  exact H0.
  Hoare_assignment_rule.
Qed.


(*
Votre programme décoré
  {{True}}
  {{x+x+x = 3*x }}
    y ::= x;;
    {{x+x+y = 3*x}}
    y ::= x + x + y
  {{y = 3*x}}.                        
*)

Lemma ex32 (x y : nat) :
  {{True}}
    y ::= x;;
    y ::= x + x + y
  {{y = 3*x}}.                        
Proof.
  Hoare_sequence_rule with (x+x+y = 3*x).
  Hoare_consequence_rule_left with (x+x+x = 3*x ).
  Impl_Intro.
  lia.
  Hoare_assignment_rule.
  Hoare_assignment_rule.
Qed.


(*
Votre programme décoré
{{x > 1}}
    a ::= 1;;  

    y ::= x;;
    {{y+a > 0 /\ y+a > x}}
    y ::= y + a
  {{y > 0 /\ y > x}}.
*)

Lemma ex33 (x y a : nat) :
  {{x > 1}}
    a ::= 1;;  
    y ::= x;;
    y ::= y + a
  {{y > 0 /\ y > x}}.
Proof.
  Hoare_sequence_rule with (y+a > 0 /\ y+a > x).
  Hoare_sequence_rule with (x+a > 0 /\ x+a > x).
  Hoare_consequence_rule_left with ( x+1 > 0 /\ x+1 > x).
  Impl_Intro.
  lia.
  Hoare_assignment_rule.
  Hoare_assignment_rule.
  Hoare_assignment_rule.
Qed.
  
(** [] *)

(** * La règle de la conditionnelle *)

(** **** Exercice : (cond) *)
(** En utilisant la règle de la conditionnelle, prouvez la validité des triplets
    ci-dessous
 *)

(** Il peut arriver d'avoir besoin de transformer une expression booléenne en
    une proposition. La tactique [bool2Prop] réalise cette opération.
*)
(*
Votre programme décoré
  {{True}}
    If (x <=? y)
    Then
    {{True /\ x <= y}}
    {{(y - x + x = y) \/ (y - x + y = x)}} 
    z ::= y - x
    {{z = y - x}}.
    {{(z + x = y) \/ (z + y = x)}}.
    Else 
    {{True /\ x > y}}
    {{(x - y + x = y) \/ (x - y + y = x)}}
    z ::= x - y
    {{(z + x = y) \/ (z + y = x)}}.
    Fi
    {{(z + x = y) \/ (z + y = x)}}.
*)

Lemma ex41 (x y z : nat) :
  {{True}}
    If (x <=? y)
    Then z ::= y - x
    Else z ::= x - y
    Fi
    {{(z + x = y) \/ (z + y = x)}}.
Proof.
  Hoare_if_rule.
  - Hoare_consequence_rule_left with (y-x+x=y \/ y - x + y =x).
    + Impl_Intro. And_Elim_2 in H. bool2Prop in H0.  lia.
    + Hoare_assignment_rule.
  - Hoare_consequence_rule_left with ((x - y + x = y) \/ (x - y + y = x)).
    + Impl_Intro. And_Elim_2 in H. bool2Prop in H0. lia.
    + Hoare_assignment_rule.
Qed.


(* Pour cet exercice, il est plus pratique d'utiliser la règle alternative de la 
   conditionnelle ci-dessous

               {{ R1}}c1{{ Q}}             {{ R2}}c2{{ Q}}
     --------------------------------------------------------------
      {{ (b -> R1) /\ (~b -> R2)}} If b Then c1 Else c2 Fi {{ Q}}
 
  la décoration du programme est plus facile à faire avec la séquence. 

Votre programme décoré

 {{True}}
    {{x+1=x+1}}
    a ::= x + 1;;
    {{a=x+1}}
    If (a - 1 =? 0)
    Then 
    {{1 = x + 1}}.
    y ::= 1
    {{y = x + 1}}.
    Else 
    {{a = x + 1}}.
    y ::= a 
    {{y = x + 1}}.
    Fi
    {{y = x + 1}}.
*)

Lemma Ex42 (a x y : nat) :
  {{True}}
    a ::= x + 1;;
    If (a - 1 =? 0)
    Then y ::= 1
    Else y ::= a 
    Fi
    {{y = x + 1}}.
Proof.
  Hoare_sequence_rule with (a=x+1).
  Hoare_consequence_rule_left with (x+1=x+1).
  Hoare_assignment_rule.
  Hoare_if_rule.
  - Hoare_consequence_rule_left with (1 = x + 1).
    +Impl_Intro. bool2Prop in H. And_Elim_2 in H. lia.
    +Hoare_assignment_rule.
  - Hoare_consequence_rule_left with (a = x +1).
    +Impl_Intro. bool2Prop in H. And_Elim_2 in H. lia.
    +Hoare_assignment_rule.
Qed.

(* [] *)

(** * La règle de la boucle *)

(**

     La règle de la boucle est 

            {{I /\ b}c{{I}}
   ----------------------------------
    {{I}}While b Do c Od{{I /\ ~b }}
*)
(** L'assertion I est un invariant pour le corps c de la boucle. À condition que
    le booléen b soit vrai, si l'assertion I est vraie avant l'exécution de c et
    que c termine alors l'assertion I est vraie après l'exécution de c.
     
    À chaque exécution de la boucle, l'assertion I est vraie et aussi avant la
    boucle, l'assertion I est vraie et après la boucle l'assertion I est vraie.
*)
(**
   La difficulté de la règle de la boucle est de trouver un invariant I
   adéquat. En effet, généralement on veut vérifier un triplet de la forme

     {{P}}While b Do c Od{{Q}}
     
   où P et Q sont des assertions qui ne sont généralement pas celles que l'on
   peut utiliser directement pour appliquer la règle de la boucle.

   On a besoin de trouver un invariant I tel que :
      * |- P -> I
      * |- (I /\ ~b) -> Q
      * |- {{I}}While b Do c Od{{I /\ ~b }}

  soient des séquents valident.
*)
(** Il n'existe malheureusement pas de méthode systématique pour trouver un
    invariant de boucle. Pour découvrir un invariant, il faut bien comprendre le
    calcul que le corps de la boucle effectue. Un invariant utilisable exprime
    l'intention du calcul effectué dans la boucle. Il donne une relation entre
    les variables manipulées dans le corps de la boucle. Cette relation est
    préservée tout au long de l'exécution de la boucle, même lorsque le contenu
    des variables est modifié.

    À partir d'un triplet de la forme {{P}}While b Do c Od{{Q}} comment trouver
    un invariant I ? En partant de la postcondition Q, il faut remonter jusqu'à
    la précondition P en insérant un invariant I.

    Il est souvent difficile de savoir par où commencer. Une marche à suivre qui
    permet d'avoir une première idée est la suivante :
    
    1. Trouver une assertion I dont vous espérez qu'elle soit un bon invariant
    2. Essayez de montrer que |- P -> I et |- I /\ b -> Q sont des séquents
       valides. Si vous y arrivez, passez au point 3 sinon recommencez au point 1.
    3. Faites remonter I au travers du corps de la boucle en décorant le
       programme c. Vous obtenez une formule I'.
    4. Essayez de prouver que le séquent |- I /\ b -> I' est valide. On montre
       ainsi que I est un invariant. Si vous y arrivez, passez au point 5 sinon
       recommencez au point 1. 
    5. Vous pouvez maintenant mettre I comme décoration au dessus de la boucle
       et mettre P comme décoration au dessus de I. Vous obtenez un programme
       décoré qui comprend un bon invariant I qui vous permet de prouver la
       validité de votre triplet.

*)

(** **** Exercice : (while) *)
(** Décorez le programme ci-dessous en trouvant le bon invariant de boucle et
    prouvez le validité du triplet.  *)

(*
Votre programme décoré
{{ n=m }} 
  {{n+0=m}}
  res ::= 0;; 
  {{n+res=m}}
  While negb (n =? 0)
  Do 
    {{(n+res=m/\ negb (n =? 0)) = true}}
    {{n - 1  + res + 1 = m}}
    res ::= res+1;;
    {{n-1 + res =m}} 
    n   ::= n-1 
    {{n+res=m}}
  Od 
  {{(n+res =m /\ negb (n =? 0)) = false}}
{{res = m}}.
 *)

Lemma ex51(m n res : nat) :
{{ n=m }} 
  res ::= 0;; 
  While negb (n =? 0)
  Do 
    res ::= res+1;; 
    n   ::= n-1 
  Od 
{{res = m}}.
Proof.
  Hoare_sequence_rule with (n+res=m).
  Hoare_consequence_rule_left with (n+0=m).
  lia.
  Hoare_assignment_rule.
  Hoare_consequence_rule_right with (n+res=m /\  (negb (n =? 0) = false)).
  Hoare_while_rule.
  Hoare_sequence_rule with ((n-1) + res =m).
  Hoare_consequence_rule_left with ((n - 1)  + (res + 1) = m).
  Impl_Intro. bool2Prop in H. lia.
  Hoare_assignment_rule.
  Hoare_assignment_rule.
  Impl_Intro. bool2Prop in H. lia.
Qed.


(** 

    Un exemple un peu plus significatif. Le programme ci-dessous calcule une
    racine carrée entière de façon naïve. Décorez ce programme et prouvez que la
    triplet est valide.
 *)

(* Dans cette preuve, on doit se "trimbaler" le (X = m) car autrement on a rien
   pour prouver (X=m -> 0*0 <= m) qui vient de l'application des règles de
   l'affectation, de la séquence et de la boucle. Il suffit de commencer par
   appliquer ces règles pour ce rendre compte du problème.

 *)

(*
Votre programme décoré
  {{ X=m }}
  {{X=m /\ m>=0}}
    Z ::= 0;;
    {{X=m /\ Z*Z<=m }}
    While ((Z+1)*(Z+1) <=? X) Do
    {{X=m /\ ((Z+1)*(Z+1) <= m}}
      Z ::= Z+1
      {{X=m /\ Z*Z<=m }}
    Od
    {{ X=m /\ Z*Z<=m  /\ (((Z+1)*(Z+1) <=? X) = false)}}
  {{ Z*Z<=m /\ m<(Z+1)*(Z+1) }}.
*)

Lemma ex52(X m Z : nat) :
  {{ X=m }}
    Z ::= 0;;
    While ((Z+1)*(Z+1) <=? X) Do
      Z ::= Z+1
    Od
    
  {{ Z*Z<=m /\ m<(Z+1)*(Z+1) }}.
Proof.
  Hoare_sequence_rule with (X=m /\ Z*Z <= m).
  Hoare_consequence_rule_left with (X=m /\ m >= 0). 
  lia.
  Hoare_assignment_rule.
  Hoare_consequence_rule_right with ((X=m /\ Z*Z<=m ) /\ (((Z+1)*(Z+1) <=? X) = false)).
  Hoare_while_rule.
  Hoare_consequence_rule_left with (X=m /\ ((Z+1)*(Z+1) <= m)).
  Impl_Intro. bool2Prop in H. lia.
  Hoare_assignment_rule.
  Impl_Intro. bool2Prop in H. lia.
Qed.
