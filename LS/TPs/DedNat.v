(** * Note préliminaire *)
(** Pour faciliter votre travail, si vous avez besoin d'une sortie papier, vous
   pouvez obtenir un fichier pdf à partir du source Coq que vous être en train
   de lire. Pour cela vous devez avoir installé LaTeX qui est un logiciel de
   composition de textes scientifiques.

*)
(** Pour générer un fichier LaTeX à partir du fichier Coq, utilisez la commande :

    << coqdoc --latex -p "\usepackage[french]{babel}" -utf8 DedNat.v >> $\\$

Ce qui crée un fichier LaTeX, << DedNat.tex >>.

Puis la commande ci-dessus crée un fichier pdf :

    << pdflatex DedNat.tex >> $\\$ crée le fichier << DedNat.pdf >>.  *)

(** 
    Si vous n'avez pas installé LaTeX, vous pouvez aussi générer un fichier html
    avec la commande 

    << coqdoc --html --no-index -utf8 DedNat.v >>
*)

(* ################################################################# *)
(** * À propos de Coq *)

(** Le système Coq est un assistant de preuves interactif basé sur une
    logique qui s'appelle le _calcul des constructions    
    inductives_. *)                  

(** Cet outil permet d'écrire des programmes, de décrire les
    propriétés de ces programmes et de prouver ces propriétés. Les
    programmes sont écrits dans un style fonctionnel. *)

(** La logique du calcul des constructions inductives utilise de façon
    fondamental l'équivalence entre preuve et calcul. Ce qui permet
    d'extraire des programmes prouvés à partir des preuves faites dans
    le système Coq. *)

(** La programmation fonctionnelle permet de se concentrer sur la
    relation entre les entrées et les sorties d'un calcul. On peut
    voir cela comme une réalisation concrète d'une fonction
    mathématique. Cela permet, à la fois, de prouver formellement la
    correction d'un programme et rationalise le raisonnement informel
    à propos du comportement d'un programme. *)

(** Concernant la programmation, la programmation fonctionnelle
    insiste sur l'utilisation des fonctions comme des valeurs qui
    peuvent être données en arguments ou retournées par des fonctions
    aussi bien que d'être stockées dans des structures de données. Le
    fait que les fonctions peuvent être utilisées comme des données
    permet une programmation avec un grand pouvoir d'expression. Cela
    permet d'utiliser simplement les types algébriques de données, le
    filtrage, des systèmes de typage sophistiqués. *)

(** * Les propositions et la déduction naturelle *)

Require Import NaturalDeduction.
(**
Vous devez récupérer sur le site du cours le fichier NaturalDeduction.v et le
compiler avec la commande :
<< coqc NaturalDeduction.v >>
*)



(** L'objectif de ce premier TP est de manipuler les règles de la déduction
   naturelle pour la logique des propositions. Il s'agit d'une première approche
   pour exprimer les propriétés des programmes. Dans la suite, on verra une
   autre logique dont le pouvoir d'expression est plus grand, la logique du
   premier ordre ou logique des prédicats, qui suffit dans la plupart des cas
   pour exprimer les propriétés des programmes. Il y a aussi des logiques dites
   d'ordre supérieur, mais cela n'entre pas dans le cadre de ce cours. *)

(** * Les propositions *)

(** Les propositions sont des phrases déclaratives au sens où elles peuvent être
   déclarées, au moins en principe, comme étant vraies ou fausses. Par exemple,
   
    - La somme des nombres 3 et 5 vaut 8.  
    - Jeanne a réagi violemment aux accusations de Jacques.  
    - Tous les entiers naturels plus grand que 2 s'écrivent comme la somme de
      deux nombres premiers.  
    - Tous les martiens aiment le poivron sur leur pizza.  
    - Die Würde des Menschen ist unantastbar.

   La vérification de la vérité ou non de ces phrases est plus ou moins
   problématique.
 *)

(** **** Exercice : (prop_sentence) *)
(** Pour chaque exemple de phrase ci-dessus expliquez comment vous pouvez
    décider de la vérité ou non de la phrase. Détaillez les problèmes que cela
    pose.
 *)

(* Remplir ici. *)
(** 
  - Faire l'addition 3+5 et comparer avec 8.
  - Pas de raisonnement. Soit c'est vrai soit c'est faux. Proposition simple.
  - On fait une preuve par récurrence. Ou donner un contre exemple si on pense que c'est faux.
  - Pas de raisonnement. Soit c'est vrai, soit c'est faux. Il suffit qu'un seul ne l'aime pas. Proposition simple.
  - Langue pas connu. Impossible à évaluer. (C'est un fait.)
*)

(** **** Exercice : (not_prop_sentence) *)
(** Donnez trois exemples de phrases qui ne sont pas déclaratives (qui ne sont
    donc pas des propositions).  
 *)

(* Remplir ici. *)
(** 
  - Fais ça !
  - Le ciel est-il bleu ?
  - Aimes-tu les poivrons sur ta pizza ?
 *)

(** Nous sommes intéressés par des propositions qui expriment le comportement
   des programmes et nous voulons vérifier que ces propositions sont vraies ou
   fausses. 

   Les propositions permettent d'exprimer un ensemble suffisamment large de
   propriétés d'un programme. La stratégie pour raisonner à propos de ces
   propriétés est de considérer certaines propositions comme non décomposables,
   on les dit atomiques, par exemple "le nombre 5 est impair". On attribue alors
   un symbole à ces propositions atomiques, par exemple on peut désigner par P
   la proposition précédente.

   Voici des exemples de propositions atomiques :
   
   - [P] : "Il pleut"
   - [Q] : "Je prend mon parapluie"
   - [R] : "Je serai mouillé"

   À partir des propositions atomiques, on construit des propositions plus
   complexes en les composants avec des connecteurs (ou opérateurs) logiques :

   - [~]  : la négation, [~P] exprime "Il ne pleut pas"
   - [\/] : la disjonction, étant donné [P] et [Q] on veut établir qu'au moins l'une des
            deux propositions est vraie, [P \/ Q] exprime "Il pleut ou je prend mon
            parapluie". Remarquez que la disjonction n'exclue pas que les deux
            propositions [P] et [Q] soient vraies (ce qui n'est pas l'usage de "ou" de la
            langue usuelle qui traduit plus souvent un "ou exclusif" où seule l'une des
            deux propositions est vraie.)
   - [/\] : la conjonction, étant donnée [P] et [Q] on veut établir des les deux
            propositions sont vraies, [P /\ Q] exprime "Il pleut et je prend mon
            parapluie".
   - [->] : l'implication, permet d'exprimer une implication entre deux
            propositions, [P -> R] exprime "s'il pleut alors je serai mouillé", [R] est une
            conséquence logique de [P]. On dit que [P] est l'hypothèse et [R] est la
            conclusion.     

  Maintenant on peut construire des propositions complexes en utilisant
  plusieurs fois les connecteurs logiques. La proposition "S'il pleut et je
  prend mon parapluie alors je ne serai pas mouillé" se note par :
        
     [(P /\ Q) -> (~R)]

  On obtient ce que l'on appelle une formule propositionnelle.

  Remarquez que pour bien noter la proposition il est nécessaire d'utiliser des
  parenthèses. Les règles de précédence pour les connecteurs logiques sont les
  suivantes :
    - [~] précède [/\] et [\/]
    - [/\] et [\/] précèdent [->] 
    - [->] est associatif à droite, [P -> Q -> R] signifie [P -> (Q -> R)]
 *)

(** **** Exercice : (prop_formula) *)
(** Donnez les notations sous forme de formules logiques des propositions
    suivantes :

    - 1. Si le soleil brille aujourd'hui alors il ne brillera pas demain
    - 2. Si le baromètre descend alors il va pleuvoir ou neiger
    - 3. Pas de chaussures, pas de chemise, pas de service
    - 4. Si une requête est faite, alors soit elle finira par être acceptée, soit
       le processus de requête ne pourra jamais progresser.
    - 5. Si M. Dupond a installé un chauffage central alors il a vendu sa voiture
       ou il n'a pas payé son hypothèque.
    - 6. A datagram whose version number is not 4 MUST be silently
       discarded. (extrait RFC 1122).
    - 7. Si Jean a rencontré Jeanne hier alors ils ont bu un café ensemble ou ils
       se sont promenés dans le parc.

 *)

(* Remplir ici. *)
(** 
  - [P -> (~Q)] avec [P] le soleil brille aujourd'hui, [Q] il brillera demain
  - [P -> (Q \/ R)] [P] le baromètre descend, [Q] il va pleuvoir et [R] il va neiger
  - [(~P),(~Q),(~R)] avec P chaussures, Q chemise, R service
  - [P -> (Q \/ ~R)] avec P une requête est faite, Q elle finira par être accepté, R le proccessus peut progresser
  - [P -> (Q \/ ~R)] avec P M. Dupond a installé un chauffage central, Q il a vendu sa voiture, R il a payé son hypothèque
  - [~P -> Q] avec P A datagram whose version number is 4, Q must be silently discarded.
  - [P -> (Q \/ R)] P si jean a rencontré jeanne hier, Q ils ont bu un café, R ils se sont promenés dans le parc.
(** * La logique (ou le calcul) des propositions *)

(** Maintenant que l'on sait exprimer des propositions complexes sous la forme
    de formules logiques, l'objectif est de raisonner de façon valide afin de
    montrer des propriétés. Par exemple, considérons le raisonnement suivant :
    
    "S'il pleut et que je n'ai pas pris mon parapluie alors je serai mouillé. Je
    ne suis pas mouillé. Il pleut. J'ai donc pris mon parapluie."
  
    Un ensemble de règles qui permet d'établir la validité ou non d'un tel
    raisonnement est un calcul logique (ou un système déductif). Les calculs que
    l'on effectue sont des calculs symboliques. Les règles d'un calcul logique
    peuvent s'écrire de différentes façons. Il y a trois façon d'écrire les
    règles d'un calcul logique, le système de Hilbert, le calcul des séquents et
    la déduction naturelle. Nous utilisons ce dernier calcul pour le cours.

    La déduction naturelle est un ensemble de règles de calcul qui permettent de
    raisonner à propos de propositions et d'obtenir une conclusion à partir
    d'hypothèses.

    Si à partir des formules [P1, P2, ..., Pn] on veut obtenir la formule [Q]
    par application des règles de calcul on note cela

             [P1, P2, ..., Pn |- Q]

    Cette notation s'appelle un séquent. Un séquent est valide si on arrive
    effectivement à obtenir la formule [Q] à partir de l'application des règles
    de calcul sur les formules [P1, P2, ..., Pn]. La suite d'applications de
    règles pour obtenir la formule [Q] s'appelle une dérivation ou une preuve.

    Nous avons vu en cours les règles de calcul pour la déduction naturelle. Ces
    règles vous sont fournies sous la forme de tactiques Coq dans le module
    [NaturalDeduction] importé au début de ce fichier.

    Pour la conjonction [/\] : - tactique pour la règle d'introduction :
    [And_Intro] - tactiques pour les règles d'élimination : [And_Elim_1],
    [And_Elim_2], ces deux tactiques s'appliquent sur une hypothèse sous la
    forme [And_Elim_{1,2} in H] où [H] est une hypothèse du contexte.

    Pour la disjonction [\/] : - tactiques pour les règles d'introduction :
    [Or_Intro_1], [Or_Intro_2] - tactique pour la règle d'élimination :
    [Or_Elim], cette tactique s'applique sur une hypothèse sous la forme
    [Or_Elim in H] où [H] est une hypothèse du contexte.

    Pour l'implication [->] : - tactique pour la règle d'introduction :
    [Impl_Intro] - tactique pour la règle d'élimination : [Impl_Elim], cette
    tactique s'applique sur deux hypothèses sous la forme [Impl_Elim in H1 and
    H2] où [H1] et [H2] sont des hypothèses du contexte.
    
    Pour la négation [~] : - tactique pour la règle d'introduction : [Not_Intro]
    - tactique pour la règle d'élimination : [Not_Elim], cette tactique
    s'applique sur deux hypothèses sous la forme [Not_Elim in H1 and H2] où [H1]
    et [H2] sont des hypothèses du contexte.

    Attention, dans le cours il n'y a pas de règle pour la négation, mais des
    règles pour l'introduction et l'élimination de "faux".  La règle
    d'élimination de la négation correspond à la règle d'introduction de "faux"
    et la règle d'introduction de la négation est la règle

<< 
                         [P] 
                          .  
                          .
                          .  
                         _|_ 
                      ------- 
                         ~P 
>> 

   Enfin, une règle supplémentaire est fournie qui permet de faire une preuve
   par contradiction. Cette règle est

<< 
                         [~P]
                           .  
                           .
                           .
                          _|_ 
                        ------- 
                           P 
>> 

   et la tactique correspondante est : [PBC] (Proof by Contradiction).

 *)

(** **** Exercice : (prop_parenthesis) *)
(** Utilisez les précédences des connecteurs logiques pour parenthéser
    complètement les expressions ci-dessous :
    
    - 1. [(~P /\ Q) -> R]
    - 2. [P -> Q -> R -> S \/ T]
    - 3. [P /\ Q \/ R]
    - 4. [P \/ Q -> ~P /\ R]

*)

(* Remplir ici. *)
(**
    - 1. [((~P) /\ Q)) -> R]
    - 2. [P -> (Q -> (R -> (S \/ T)))]
    - 3. [((P /\ Q) \/ R)]
    - 4. [(P \/ Q) -> ((~P) /\ R)]

 *)

(** **** Exercice : (prop_proof) *)
(** Pour chaque séquent ci-dessous, écrivez une phrase en français qui l'énonce,
    prouvez sa validité et dessinez sur une feuille l'arbre de la preuve obtenue :
    
    - 1. [P |- Q -> (P /\ Q)]
    - 2. [(P /\ Q) /\ R, S /\ T |- Q /\ S]
    - 3. [P \/ Q |- Q \/ P]
    - 4. [(P \/ Q) \/ R |- P \/ (Q \/ R)]
    - 5. [P -> (Q -> R), P -> Q |- P -> R]
    - 6. [P \/ Q, ~Q |- P]
    - 7. [P -> (Q -> R), P, ~R |- ~Q]
 *)

    (* Remplir ici. *)
(** 
    - 1. Si on a P, alors on a Q implique P et Q. 
    - 2. Si on a P et Q et R, S et T alors on a Q et S
    - 3. Si on a P ou Q alors on a Q ou P.
    - 4. Si on a (P ou Q) ou R alors on a P ou (Q ou R)
    - 5. Si on a P implique (Q implique R), P implique Q alors on a P implique R
    - 6. Si on a P ou Q, ~Q alors on a P.
    - 7. Si on a P implique (Q implique R), P, ~R alors on a ~Q.
 *)
*)
Lemma exproof1 (P Q R : Prop) : P -> Q -> (P /\ Q).
Proof.
  Impl_Intro.
  Impl_Intro.
  And_Intro.
  exact H.
  exact H0.
Qed.

Lemma exproof2 (P Q R S T: Prop) : (P /\ Q) /\ R -> S /\ T -> Q /\ S.
Proof.
  Impl_Intro.
  Impl_Intro.
  And_Elim_1 in H0.
  And_Elim_1 in H.
  And_Elim_2 in H2.
  And_Intro.
  exact H3.
  exact H1.
Qed.


Lemma exproof3 (P Q : Prop) : P \/ Q -> Q \/ P.
Proof.
  Impl_Intro.
  Or_Elim in H.
  Or_Intro_2.
  exact H.
  Or_Intro_1.
  exact H.
Qed.

Lemma exproof4 (P Q R : Prop) : (P \/ Q) \/ R -> P \/ (Q \/ R).
Proof.
  Impl_Intro.
  Or_Elim in H.
  Or_Elim in H.
  Or_Intro_1.
  exact H.
  Or_Intro_2.
  Or_Intro_1.
  exact H.
  Or_Intro_2.
  Or_Intro_2.
  exact H.
Qed.

Lemma exproof5 (P Q R : Prop) : (P -> (Q -> R)) -> (P -> Q) -> (P -> R).
Proof.
  Impl_Intro.
  Impl_Intro.
  Impl_Intro.
  Impl_Elim in H and H1.
  exact H2.
  Impl_Elim in H0 and H1.
  exact H3.
Qed.

Lemma exproof6 (P Q : Prop) : P \/ Q -> ~Q -> P.
Proof.
  Impl_Intro.
  Impl_Intro.
  Or_Elim in H.
  exact H.
  PBC.
  Not_Elim in H0 and H.
  exact H2.
Qed. 

Lemma exproof7 (P Q R : Prop) :(P -> (Q -> R)) -> (P) -> (~R) -> (~Q).
Proof.
  Impl_Intro.
  Impl_Intro.
  Impl_Intro.
  Not_Intro.
  Impl_Elim in H and H0.
  Not_Elim in H1 and  H3.
  exact H4.
  exact H2.
Qed.
  

(** **** Exercice : (prop_formalize) *)
(** Traduisez le raisonnement suivant sous la forme d'un séquent et prouvez la
    validité de ce séquent.  

    "S'il pleut et que je n'ai pas pris mon parapluie alors je serai mouillé. Je
    ne suis pas mouillé. Il pleut. J'ai donc pris mon parapluie."

 *)

    (* Remplir ici. *)
(** 
  - [(P /\ ~Q) -> R, ~R, P |- Q ] 
*)

(* J'y arrive pas.*)
Lemma expluie (P Q R: Prop) : P /\ ~Q -> R -> (~R -> (P -> Q)).
Proof.
  Impl_Intro.
  Impl_Intro.
  Impl_Intro.
  Impl_Intro.
  Not_Elim in H1 and H0.
  PBC.
  exact H3.
Qed.

(** Dans certains cas, on a besoin de faire une supposition temporaire,
    d'ajouter une hypothèse au contexte de la preuve pour pouvoir avancer dans
    la preuve. Cela se fait avec la tactique [assume].

    Par exemple, prouvons le séquent

    [(P /\ Q) -> R, P, Q |- R]

    On le transforme en un théorème :

    [Lemma exhyp (P Q R: Prop) : ((P /\ Q) -> R) -> (P -> (Q -> R)).]
*)

Lemma exhyp (P Q R: Prop) : ((P /\ Q) -> R) -> (P -> (Q -> R)).
Proof.
  Impl_Intro.
  Impl_Intro.
  Impl_Intro.
  (* À cet endroit de la preuve, on est bloqué. On voudrait bien "attraper" le R
     qui est dans la conclusion du "implique" de l'hypothèse H dans notre
     contexte. Pour cela on peut supposer que l'on a (P/\Q) et le prouver par la
     suite. Ceci se fera sans peine car on a déjà P et Q dans comme hypothèses
     dans notre contexte.  *)
  assume (P /\ Q).
  (* le reste de la preuve suit sans difficulté *)
  Impl_Elim in H and H2.
  exact H3.
  And_Intro.
  exact H0.  
  exact H1.
Qed.

(** **** Exercice : (prop_transform) *)
(** Vous avez sans doute déjà rencontré les connecteurs logiques et vous savez
    peut être que l'on peut passer d'un connecteur à l'autre via certaines
    transformations. Par exemple, on a les égalités

    - 1. [~(P \/ Q) = ~P /\ ~Q]
    - 2. [~(P /\ Q) = ~P \/ ~Q]
    - 3. [(P -> Q) = ~P \/ Q]

    Une façon de vérifier que l'on a bien ces égalités est d'écrire les tables
    de vérité des expressions figurant de chaque coté du signe égal et de
    constater qu'on obtient les mêmes tables. Ce passage par les tables de
    vérités est une approche par la sémantique des formules propositionnelles. 

    On peut prouver formellement que l'on a bien ces égalités. Pour cela on
    transforme l'égalité en deux implications, l'un qui va de la gauche vers la
    droite et l'autre qui va de la droite vers la gauche par rapport au signe
    égal. Par exemple pour [~(P \/ Q) = ~P /\ ~Q], on écrit la formule
    
      [(~(P \/ Q) -> ~P /\ ~Q) /\ (~P /\ ~Q -> ~(P \/ Q))]

    que l'on preuve ensuite.

    Prouvez les trois égalités données ci-dessus. Pour la dernière égalité vous 
    allez avoir besoin de prouver un résultat intermédiaire qui est le théorème
    
    [|- P \/ ~P]

    qui est la règle du "tiers exclu".

*)


(* Remplir ici. *)
(** 
  - 1. [(~(P\/Q) -> ~P /\ ~Q) /\ (~P /\ ~Q -> ~(P \/ Q))] 
  - 2. [(~(P/\Q) -> ~P \/ ~Q) /\ (~P \/ ~Q -> ~(P /\ Q))] 
  - 3. [((P -> Q) -> ~P \/ Q) /\ (~P \/ Q -> (P -> Q))]
*)

Lemma ex1 (P Q R: Prop) :  (~P /\ ~Q -> ~(P \/ Q) )/\ (~(P\/Q) -> ~P /\ ~Q) .
Proof.
  And_Intro.
  Impl_Intro.
  Not_Intro.
  And_Elim_1 in H.
  Or_Elim in H0.
  Not_Elim in H1 and H0.
  exact H2.
  And_Elim_2 in H.
  Not_Elim in H2 and H0.
  exact H3.
  Impl_Intro.
  And_Intro.
  Not_Intro.
  assume (P\/ Q).
  Not_Elim in H and H1.
  exact H2.
  Or_Intro_1.
  exact H0.
  Not_Intro.
  assume(P\/Q).
  Not_Elim in H and H1.
  exact H2.
  Or_Intro_2.
  exact H0.
Qed.

Lemma ex2 (P Q R : Prop) :  (~P \/ ~Q -> ~(P /\ Q)) /\ ((~(P/\Q) -> ~P \/ ~Q)).
Proof.
  And_Intro.
  Impl_Intro.
  Not_Intro.
  Or_Elim in H.
  And_Elim_1 in H0.
  Not_Elim in H and H1.
  exact H2.
  And_Elim_2 in H0.
  Not_Elim in H and H1.
  exact H2.
  Impl_Intro.
  PBC.
  assume(P/\Q).
  Not_Elim in H and H1.
  exact H2.
  And_Intro.
  PBC.
  assume (~P\/~Q).
  Not_Elim in H0 and H2.
  exact H3.
  Or_Intro_1.
  exact H1.
  PBC.
  assume(~P\/~Q).
  Not_Elim in H0 and H2.
  exact H3.
  Or_Intro_2.
  exact H1.
Qed.

Lemma tiers_exclu (P : Prop) : ( P \/ ~P).
Proof.
  PBC.
  assume(P \/ ~P).
  Not_Elim in H and H0.
  exact H1.
  Or_Intro_2.
  Not_Intro.
  assume (P \/ ~P).
  Not_Elim in H and H1.
  exact H2.
  Or_Intro_1.
  exact H0.
Qed.


Lemma ex3 (P Q R : Prop) :  (~P \/ Q -> (P -> Q)) /\ ((P -> Q) -> ~P \/ Q).
Proof.
  And_Intro.
  intros.
  Or_Elim in H.
  PBC.
  Not_Elim in H and H0.
  exact H2.
  exact H.
  intros.
  (*A finir*)
  Or_Intro_1.
  Not_Intro.
  Impl_Elim in H and H0.
  assume (~P \/ Q).
  Or_Elim in H2.
  Not_Elim in H2 and H0.
  exact H3.
