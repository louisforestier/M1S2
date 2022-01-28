(** On met en place l'environnement usuel pour les exercices. N'oubliez pas de
    compiler *)
    Require Import NaturalDeduction.


    (** * Sur la logique *)
    
    (** **** Exercice : (prop_formula) *)
    (** Donnez les notations sous forme de formules logiques des propositions
        suivantes :
        
        - 1. Si le chat voit l'oiseau, il le mangera
        - 2. Aujourd'hui il va pleuvoir ou faire beau, mais pas les deux.
        - 3. Le costume d'Arlequin est jaune et vert.

    *)

(*
Réponses :

    - 1. [P -> Q] avec P le chat voit l'oiseau et Q il mange l'oiseau.
    - 2. [P \/ Q /\ ~(P /\ Q)] avec P il va pleuvoir aujourd'hui et Q il va faire beau aujourd'hui.
    - 3. [P /\ Q] avec P le costume d'Arlequin est jaune et Q le costume d'Arlequin est vert.
*)
    
    (** **** Exercice : (prop_proof) *)
    
    (** Prouvez la validité des séquents ci-dessous :
    
        1. [P -> Q, P -> ~Q |- ~P]
        2. [P \/ Q |- R -> (P \/ Q) /\ R]
        3. [(P /\ Q) \/ (Q /\ R) |- Q /\ (P \/ R)]
     *)
    
Theorem  prop_proof1  (P Q : Prop): P -> Q -> P -> ~Q -> ~P .
Proof.
    intros.
    PBC.
    Not_Elim in H2 and H0.
    exact H4.
Qed.

Theorem prop_proof2 (P Q R : Prop) : P \/ Q -> R -> (P \/ Q) /\ R.
Proof.
    intros.
    And_Intro.
    exact H.
    exact H0.
Qed.

Theorem prop_proof3 (P Q R : Prop): (P /\ Q) \/ (Q /\ R) -> Q /\ (P \/ R).
Proof.
    intros.
    
    Or_Elim in H.
    And_Elim_2 in H.
    And_Intro.
    exact H0.
    Or_Intro_1.
    And_Elim_1 in H.
    exact H1.
    And_Elim_1 in H.
    And_Intro.
    exact H0.
    And_Elim_2 in H.
    Or_Intro_2.
    exact H1.
Qed.

    
    (** **** Exercice : (prop_deduct) *)
    (**
       1. Formalisez sous la forme d'un séquent le texte ci-dessous :
       
         Si le chat voit un oiseau, il miaule. Le chat n'a pas
         miaulé. Aucun oiseau n'a été vu par le chat.
    
       2. Prouvez la validité du séquent obtenu.
    
       3. Ce séquent est tellement couramment utilisé qu'il peut prendre le statu
       d'une règle (c'est le modus tollens). Écrivez cette règle.
    
     *)
    
    (* Remplir ici. *)
    (*
    1. P -> Q, ~Q |- ~P avec P le chat voit un oiseau et Q le chat miaule.
    *)
Theorem prop_deduct (P Q : Prop): P -> Q -> (~Q -> ~P).
Proof.
    intros.
    PBC.
    Not_Elim in H1 and H0.
    exact H3.    
Qed.

(*

       Q  ~Q
       -----
P       _|_
-----------------
P -> Q, ~Q |- ~P

En coq, ça devrait donner quelque chose de ce style.
Definition Modus_Tollens (p1:Prop) (p2:Prop) :=
match (p1, p2) with
    | ( P -> Q, ~Q) => ~P
    | _ => fail 1 "This is not a modus tollens."
end.
*)
