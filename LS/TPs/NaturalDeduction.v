Ltac And_Intro :=
match goal with
  | |- _ /\ _ => split
  | _         => fail 1 "Goal is not an And-formula"
end.

Ltac Or_Intro_1 :=
match goal with
  | |- _ \/ _ => left
  | _         => fail 1 "Goal is not an Or-formula"
end.

Ltac Or_Intro_2 :=
match goal with
  | |- _ \/ _ => right
  | _         => fail 1 "Goal is not an Or-formula"
end.

Ltac Impl_Intro :=
match goal with
  | |- _ -> _ => let H := fresh "H" in intro H
  | _         => fail 1 "Goal is not an Implication-formula"
end.

Ltac Not_Intro :=
match goal with
  | |- ~ _ => let H := fresh "H" in intro H
  | _      => fail 1 "Goal is not a Not-formula"
end.

Ltac Forall_Intro :=
match goal with
  | |- forall x, _ => let x := fresh x in intro x
  | _              => fail 1 "Goal is not a Forall-formula"
end.

Ltac Exists_Intro' t :=
match goal with
  | |- exists x, _ => exists t
  | _              => fail 1 "Goal is not an Exists-formula"
end.

Tactic Notation "Exists_Intro" "with" constr(t) := Exists_Intro' t.

Ltac And_Elim_1' H :=
match type of H with
  | _ /\ _ => let H0 := fresh "H" in
              let H1 := fresh "H" in
              assert (H0 := H); destruct H0 as [H0 H1]; clear H1
  | _      => fail 1 "Hypothesis is not an And-formula"
end.

Tactic Notation "And_Elim_1" "in" hyp(H) := And_Elim_1' H.

Ltac And_Elim_2' H :=
match type of H with
  | _ /\ _ => let H0 := fresh "H" in
              let H1 := fresh "H" in
              assert (H0 := H); destruct H0 as [H1 H0]; clear H1
  | _      => fail 1 "Hypothesis is not an And-formula"
end.

Tactic Notation "And_Elim_2" "in" hyp(H) := And_Elim_2' H.

Ltac And_Elim_all' H :=
match type of H with
  | _ /\ _ => let H0 := fresh "H" in
              destruct H as [H H0]; And_Elim_all' H; And_Elim_all' H0
  | _      => idtac
end.

Tactic Notation "And_Elim_all" "in" hyp(H) := And_Elim_all' H.

Ltac Or_Elim' H :=
match type of H with
  | _ \/ _  => destruct H
  | _       => fail 1 "Hypothesis is not an Or-formula"
end.

Tactic Notation "Or_Elim" "in" hyp(H) := Or_Elim' H.

Ltac Impl_Elim' H0 H1 :=
match type of H0 with
  | ?P -> _ => 
    match type of H1 with
      | P => let H2 := fresh "H" in assert (H2 := H1); apply H0 in H2
      | _ => fail 2 "Second hypothesis does not match the assumption of the first hypothesis"
    end
  | _       => fail 1 "First hypothesis is not an Implication-formula"
end.

Tactic Notation "Impl_Elim" "in" hyp(H0) "and" hyp(H1) := Impl_Elim' H0 H1.

Ltac Not_Elim' H0 H1 :=
match type of H0 with
  | ~ ?P => 
    match type of H1 with
      | P => let H2 := fresh "H" in assert (H2 := H1); apply H0 in H2
      | _ => fail 2 "Second hypothesis does not match the body of the first hypothesis"
    end
  | _       => fail 1 "First hypothesis is not a Not-formula"
end.

Tactic Notation "Not_Elim" "in" hyp(H0) "and" hyp(H1) := Not_Elim' H0 H1.

Ltac Forall_Elim' H t :=
match type of H with
  | forall x, _ => let H0 := fresh "H" in assert (H0 := H t)
  | _           => fail 1 "Hypothesis is not a Forall-formula"
end.

Tactic Notation "Forall_Elim" "in" hyp(H) "with" constr(t) := Forall_Elim' H t.

Ltac Exists_Elim' H :=
match type of H with
  | exists x, _ => let H0 := fresh "H" in assert (H0 := H); destruct H0
  | _           => fail 1 "Hypothesis is not an Exists-formula"
end.

Tactic Notation "Exists_Elim" "in" hyp(H) := Exists_Elim' H.

Require Import Classical.

Ltac PBC := let H := fresh "H" in apply NNPP; intro H.

Ltac assume P := cut P; [ let H := fresh "H" in intro H | idtac ].
