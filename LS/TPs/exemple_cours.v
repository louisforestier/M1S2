Require Import NaturalDeduction.

Variables P Q R : Prop.

Lemma Ex_cours : (P /\ Q) -> R -> Q /\ R.
Proof.
  intros.
  And_Intro.
  And_Elim_2 in H.
  exact H1.
  exact H0.
Qed.

(*Pour éviter var global on peut donner des arg*)

Lemma Ex_cours_arg( P Q R : Prop) : (P /\ Q) -> R -> Q /\ R.
Proof.
  intros.
  And_Intro.
  And_Elim_2 in H.
  exact H1.
  exact H0.
Qed.

Module ArithWithConstants.

Inductive arith : Type := 
  | Const : nat -> arith
  | Plus : arith -> arith -> arith
  | Times : arith -> arith -> arith.

  Example ex1 := Const 42.
  Example ex2 :=  Plus (Const 1) (Times (Const 2) (Const 3)).

Fixpoint size(e : arith) : nat :=
  match e with
  |Const _ => 1
  |Plus e1 e2 => 1 + size e1 + size e2
  |Times e1 e2 => 1 + size e1 + size e2
  end.

  Compute size ex1.
  Compute size ex2.

Fixpoint depth(e : arith) : nat :=
  match e with 
  |Const _ => 1
  |Plus e1 e2 => max (depth e1) (depth e2)
  |Times e1 e2 => max (depth e1) (depth e2)
  end.


  Compute depth ex1.
  Compute depth ex2.

Theorem depth_le_size : forall e, depth e <= size e.
Proof.
  induction e.

  simpl.
  reflexivity.

  simpl.
  lia.

  simpl.
  lia.

Qed.

