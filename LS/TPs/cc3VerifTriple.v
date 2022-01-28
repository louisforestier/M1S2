(** On met en place l'environnement usuel pour les exercices. N'oubliez pas de
    compiler les trois fichiers *)
    Require Import NaturalDeduction.
    Require Import Hoare.
    Require Import Lemmas.
    
    (* De quoi pouvoir travailler avec les entiers *)
    Require Import PeanoNat.
    Import Nat.
    
    (* La théorie décidable de l'arithmétique linéaire *)
    Require Import Lia.
    
    (** * Sur les triplets de Hoare *)
    
    (** **** Exercice : (hoare_seq) *)
    
    (**
       Après avoir donné la décoration du programme, prouvez le triplet de Hoare suivant.
    
     **)

    (*
    Décoration:
    {{(a = 6) /\ (b = 2)}}
        {{(((b + a) - a) = 2) /\ (((b + a) - ((b+a) - a))= 6)}}.
        b ::= b + a ;;
        {{((b - a) = 2) /\ ((b - (b - a)) = 6)}}.
        a ::= b - a ;;
        {{(a = 2) /\ ((b - a) = 6)}}.
        b ::= b - a 
    {{(a = 2) /\ (b = 6)}}.
    *)
    
    Lemma ex20 (a b : nat) :
      {{(a = 6) /\ (b = 2)}}
        b ::= b + a ;;
        a ::= b - a ;;
        b ::= b - a 
      {{(a = 2) /\ (b = 6)}}.
    Proof.
        Hoare_sequence_rule with ((a = 2) /\ ((b - a) = 6)).
        Hoare_sequence_rule with (((b - a) = 2) /\ ((b - (b - a)) = 6)).
        Hoare_consequence_rule_left with ((((b + a) - a) = 2) /\ (((b + a) - ((b+a) - a))= 6)).
        intros. And_Intro. lia. simpl. lia.
        Hoare_assignment_rule.
        Hoare_assignment_rule.
        Hoare_assignment_rule.
    Qed.
    
    (** **** Exercice : (hoare_if) *)
    (**
    
       1. Expliquez ce que fait le programme ci-dessous et dites pourquoi la postcondition
          décrit bien le résultat attendu du programme.
       2. Décorez le programme.
       3. Prouvez la validité du triplet de Hoare que vous avez formé. 
    
    *) 
    (*
        1.
        Si y est inférieur à x, alors on donne la valeur de y à z, sinon on lui donne la valeur de x.
        Par conséquent, on assigne à z le minimum entre x et y.
        La postcondition demande donc que z corresponde à ce minimum et qu'il soit forcément égale soit à x, soit à y.

        2.
        {{True}}
        If (y <? x)
        Then 
        {{(y = (min x y) /\ (y = y))}}.
        z ::= y
        {{(z = (min x y) /\ (z = y))}}.
        {{(z = (min x y) /\ (z = x \/ z = y))}}.
        Else 
        {{(x = (min x y) /\ (x = x))}}.
        z ::= x
        {{(z = (min x y) /\ (z = x))}}.
        {{(z = (min x y) /\ (z = x \/ z = y))}}.
        Fi
        {{(z = (min x y) /\ (z = x \/ z = y))}}.
    *)
    
    Lemma ex21 (x y z : nat) :
          {{True}}
          If (y <? x)
          Then z ::= y
          Else z ::= x
          Fi
          {{(z = (min x y) /\ (z = x \/ z = y))}}.
    Proof.
        Hoare_if_rule.
        Hoare_consequence_rule_left with ((y = (min x y) /\ ( y= y))).
        intros. bool2Prop in H. lia.
        Hoare_consequence_rule_right with ((z = (min x y) /\ (z = y))).
        Hoare_assignment_rule.
        intros. lia.
        Hoare_consequence_rule_left with ((x = (min x y) /\ ( x= x))).
        intros. bool2Prop in H. lia.
        Hoare_consequence_rule_right with ((z = (min x y) /\ (z = x))).
        Hoare_assignment_rule.
        intros. lia.
    Qed.
    
    (** **** Exercice : (hoare_cpy) *)
    (**
     
       Le programme ci-dessous recopie la valeur de x dans la varible y de façon
       naïve.
       
       1. Convainquez-vous que l'affirmation ci-dessus est vraie en déroulant le
       programme sur deux exemples significatifs.
    
       2. Donnez un invariant pour la boucle. Pour cela vous pouvez observer les
       valeurs des variables à chaque tour de boucle et déduisez en une quantité
       invariante définie à partir de ces valeurs.
    
       3. Décorez le programme et prouvez la validité du triplet de Hoare.
    
     *)

     (*
     1. on prend x = 5. y= 0 au début, puis on incrémente y jusqu'à ce qu'il soit égal à x.
     *)
    
    (*
    Décoration :
    {{x >= 0}}
        y ::= 0;;
        {{x>=y}}
        While (negb (x =? y))
        Do
        {{x>=y+1}}
        y ::= y + 1
        {{x>=y}}
        Od
        {{(x>=y) /\ (negb(x=?y) = false)}}
    {{x = y}}.
    *)
    
    Lemma ex22 (x y : nat) :
      {{x >= 0}}
        y ::= 0;;
        While (negb (x =? y))
        Do
        y ::= y + 1
        Od
      {{x = y}}.
    Proof.
        Hoare_sequence_rule with (x >= y).
        Hoare_assignment_rule.
        Hoare_consequence_rule_right with ((x>=y) /\ (negb (x=? y)=false)).
        Hoare_while_rule.
        Hoare_consequence_rule_left with (x>=y+1).
        intros. bool2Prop in H. lia.
        Hoare_assignment_rule. 
        intros. bool2Prop in H. lia.
    Qed. 

    
    