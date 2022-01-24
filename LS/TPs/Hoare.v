Require Export Bool.
Require Export PeanoNat.
Require Export Minus.
Export Nat.
Require Export List.
Export ListNotations.

Inductive Prog :=
  | Skip   : Prog
  | Assign : forall (ty : Type), ty -> ty -> Prog
  | Seq    : Prog -> Prog -> Prog
  | Cond   : bool -> Prog -> Prog -> Prog
  | Loop   : bool -> Prog -> Prog.

Infix "::=" := (Assign _) (at level 90).
Infix ";;" := Seq (at level 96, left associativity).
Notation "'If' b 'Then' c1 'Else' c2 'Fi'" := (Cond b c1 c2) (at level 93).
Notation "'While' b 'Do' c 'Od'" := (Loop b c) (at level 93).

(* ajout LF 26/07/19 *)
Notation "'!' b" := (negb b) (at level 70).

Fixpoint countLoops (p : Prog) : nat :=
match p with
  | Skip         => 0
  | Assign _ _ _ => 0
  | Seq c1 c2    => countLoops c1 + countLoops c2
  | Cond _ c1 c2 => countLoops c1 + countLoops c2
  | Loop _ c     => 1 + countLoops c 
end.

Inductive Hoare : Prop -> Prog -> Prop -> Prop :=
  | RCons   : forall (P' Q' P Q :Prop) (c : Prog), (P -> P') -> Hoare P' c Q' -> (Q' -> Q) -> Hoare P c Q
  | RSkip   : forall (P : Prop), Hoare P Skip P
  | RAssign : forall (ty : Type) (P : ty -> Prop) (x t : ty), Hoare (P t) (Assign ty x t) (P x)
  | RSeq    : forall (P R Q : Prop) (c1 c2 : Prog), Hoare P c1 R -> Hoare R c2 Q -> Hoare P (Seq c1 c2) Q
  | RIf     : forall (P Q : Prop) (b : bool) (c1 c2 : Prog), Hoare (P /\ b = true) c1 Q -> Hoare (P /\ b = false) c2 Q -> Hoare P (Cond b c1 c2) Q
  | RWhile  : forall (I : Prop) (b : bool) (c : Prog), Hoare (I /\ b = true) c I -> Hoare I (Loop b c) (I /\ b = false).

Notation "{{ P }} c {{ Q }}" := (Hoare P c Q) (at level 90, c at next level).

Ltac Hoare_consequence_rule' P' Q' := 
match goal with 
  | |- Hoare ?P ?Pr ?Q => 
     apply (RCons P' Q' P Q Pr); [ trivial | idtac | trivial ]
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Tactic Notation "Hoare_consequence_rule" "with" constr(P) "and" constr(Q) := Hoare_consequence_rule' P Q.

Ltac Hoare_consequence_rule_left' P' := 
match goal with 
  | |- Hoare ?P ?Pr ?Q => 
     apply (RCons P' Q P Q Pr); [ trivial | idtac | trivial ]
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Tactic Notation "Hoare_consequence_rule_left" "with" constr(P) := Hoare_consequence_rule_left' P.

Ltac Hoare_consequence_rule_right' Q' := 
match goal with 
  | |- Hoare ?P ?Pr ?Q => 
     apply (RCons P Q' P Q Pr); [ trivial | idtac | trivial ]
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Tactic Notation "Hoare_consequence_rule_right" "with" constr(P) := Hoare_consequence_rule_right' P.

Ltac Hoare_skip_rule := 
match goal with 
  | |- Hoare ?P ?Pr ?Q => 
     match Pr with 
       | Skip => first [ apply RSkip | fail 3 "Precondiction is not right" ]
       | _    => fail 2 "Goal is not a Skip-statement"
     end
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Ltac Hoare_assignment_rule :=
match goal with 
  | |- Hoare ?P ?Pr _ => 
     match Pr with 
       | Assign ?ty ?x ?t =>
          first [ is_var x | fail 3 "Left side is not a variable" ];
          pattern x;
          let Pf := match goal with
                       | |- (fun (n : ty) => Hoare _ _ ?Q) x => 
                          constr:(fun (n : ty) => Q)
                    end in
          assert (H : P -> Pf t); [ let H0 := fresh "H" in intro H0 ; first [ assumption | fail 3 "Precondiction is not right" ] | exact (RAssign ty Pf x t) ]
       | _ => fail 2 "Goal is not an assignment"
     end
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Ltac Hoare_sequence_rule' R :=
match goal with 
  | |- Hoare ?P ?Pr ?Q => 
     match Pr with 
       | Seq ?c1 ?c2 => apply (RSeq P R Q)
       | _           => fail 2 "Goal is not a sequential composition"
     end
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Tactic Notation "Hoare_sequence_rule" "with" constr(P) := Hoare_sequence_rule' P.

Ltac Hoare_if_rule :=
match goal with 
  | |- Hoare ?P ?Pr ?Q => 
     match Pr with 
       | Cond ?b ?c1 ?c2 => apply (RIf P Q b c1 c2)
       | _               => fail 2 "Goal is not an If-statement"
     end
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Ltac Hoare_while_rule :=
match goal with 
  | |- Hoare ?I ?Pr ?Q => 
     match Pr with 
       | Loop ?b ?c => match Q with
                         | I /\ b = false => apply (RWhile I b c) 
                         | _              => fail 3 "Post condition is not of the right form"
                       end
       | _          => fail 2 "Goal is not a While-statement"
     end
  | _ => fail 1 "Goal is not a Hoare formula"
end.

Lemma curry : forall (P Q R : Prop), (P -> Q -> R) -> (P /\ Q) -> R.
Proof.
tauto.
Qed.


Lemma triv1 : forall (x : bool), x = x <-> True.
Proof.
intros; split; intros; trivial.
Qed.

Lemma triv2 : true = false <-> False.
Proof.
split; intro H; [ inversion H | contradiction ].
Qed.

Lemma triv3 : false = true <-> False.
Proof.
split; intro H; [ inversion H | contradiction ].
Qed.


Ltac bool2Prop_tactic_goal :=
match goal with
  | |- context f [true = true]       => rewrite triv1; bool2Prop_tactic_goal
  | |- context f [true = false]      => rewrite triv2; bool2Prop_tactic_goal
  | |- context f [false = true]      => rewrite triv3; bool2Prop_tactic_goal
  | |- context f [false = false]     => rewrite triv1; bool2Prop_tactic_goal
  | |- context f [negb _ = true]     => rewrite negb_true_iff; bool2Prop_tactic_goal
  | |- context f [negb _ = false]    => rewrite negb_false_iff; bool2Prop_tactic_goal
  | |- context f [_ && _ = true]     => rewrite andb_true_iff; bool2Prop_tactic_goal
  | |- context f [_ && _ = false]    => rewrite andb_false_iff; bool2Prop_tactic_goal
  | |- context f [_ || _ = true]     => rewrite orb_true_iff; bool2Prop_tactic_goal
  | |- context f [_ || _ = false]    => rewrite orb_false_iff; bool2Prop_tactic_goal
  | |- context f [(_ =? _) = true]   => rewrite eqb_eq; bool2Prop_tactic_goal
  | |- context f [(_ =? _) = false]  => rewrite eqb_neq; bool2Prop_tactic_goal
  | |- context f [(_ <=? _) = true]  => rewrite leb_le; bool2Prop_tactic_goal
  | |- context f [(_ <=? _) = false] => rewrite leb_nle; bool2Prop_tactic_goal
  | |- context f [(_ <? _) = true]   => rewrite ltb_lt; bool2Prop_tactic_goal
  | |- context f [(_ <? _) = false]  => rewrite ltb_nlt; bool2Prop_tactic_goal
  | |- context f [even _ = true]     => rewrite even_spec; bool2Prop_tactic_goal
  | |- context f [even _ = false]    => rewrite <- not_true_iff_false ; rewrite even_spec; bool2Prop_tactic_goal
  | |- _                             => idtac
end.

Ltac bool2Prop_tactic_hyp H :=
match type of H with
  | context f [true = true]       => rewrite triv1 in H; bool2Prop_tactic_hyp H
  | context f [true = false]      => rewrite triv2 in H; bool2Prop_tactic_hyp H
  | context f [false = true]      => rewrite triv3 in H; bool2Prop_tactic_hyp H
  | context f [false = false]     => rewrite triv1 in H; bool2Prop_tactic_hyp H
  | context f [negb _ = true]     => rewrite negb_true_iff in H; bool2Prop_tactic_hyp H
  | context f [negb _ = false]    => rewrite negb_false_iff in H; bool2Prop_tactic_hyp H
  | context f [_ && _ = true]     => rewrite andb_true_iff in H; bool2Prop_tactic_hyp H
  | context f [_ && _ = false]    => rewrite andb_false_iff in H; bool2Prop_tactic_hyp H
  | context f [_ || _ = true]     => rewrite orb_true_iff in H; bool2Prop_tactic_hyp H
  | context f [_ || _ = false]    => rewrite orb_false_iff in H; bool2Prop_tactic_hyp H
  | context f [(_ =? _) = true]   => rewrite eqb_eq in H; bool2Prop_tactic_hyp H
  | context f [(_ =? _) = false]  => rewrite eqb_neq in H; bool2Prop_tactic_hyp H
  | context f [(_ <=? _) = true]  => rewrite leb_le in H; bool2Prop_tactic_hyp H
  | context f [(_ <=? _) = false] => rewrite leb_nle in H; bool2Prop_tactic_hyp H
  | context f [(_ <? _) = true]   => rewrite ltb_lt in H; bool2Prop_tactic_hyp H
  | context f [(_ <? _) = false]  => rewrite ltb_nlt in H; bool2Prop_tactic_hyp H
  | context f [even _ = true]     => rewrite even_spec in H; bool2Prop_tactic_hyp H
  | context f [even _ = false]    => rewrite <- not_true_iff_false in H; rewrite even_spec in H; bool2Prop_tactic_hyp H
  | _                             => idtac
end.

Tactic Notation "bool2Prop" "in" hyp(H) := bool2Prop_tactic_hyp H.
Tactic Notation "bool2Prop" := bool2Prop_tactic_goal.