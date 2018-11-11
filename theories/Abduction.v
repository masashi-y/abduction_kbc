
Declare ML Module "abduction".

Axiom proof_admitted : False.
Ltac admit_False := case proof_admitted.

Definition impl_fun1 (T : Type) (P Q : T -> Prop) : Prop :=
    forall x, P x -> Q x.

Definition impl_fun2 (T S : Type) (P Q : T -> S -> Prop) : Prop :=
    forall x y, P x y -> Q x y.

Definition ant_impl_fun1 (T : Type) (P Q : T -> Prop) : Prop :=
    forall x, P x -> Q x -> False.

Definition ant_impl_fun2 (T S : Type) (P Q : T -> S -> Prop) : Prop :=
    forall x y, P x y -> Q x y -> False.

Definition ant_impl_fun1_subj (T S : Type) (P Q : T -> Prop) (Subj : T -> S) : Prop :=
    forall F x y, P x -> Q y -> F (Subj x) -> F (Subj y)  -> False.

(*
Definition ant_impl_fun2_subj (T S : Type) (P Q : T -> S -> Prop) : Prop :=
    forall F x y z w, P x z -> Q y -> F (Subj x) -> F (Subj y)  -> False.
*)

Definition backoff_fun1 (T : Type) (P Q : T -> Prop) : Prop :=
    forall x, P x -> Q x.

Definition backoff_fun2 (T S : Type) (P Q : T -> S -> Prop) : Prop :=
    forall x y, P x y -> Q x y.

Ltac abduction :=
    ij_hello.
