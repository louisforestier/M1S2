(** * Induction : preuve par induction *)

(** Avant de commencer nous avons besoin d'importer les définitions du
    précédent chapitre. *)

Require Export LSBasics.

(** Pour que [Require Export] fonctionne, vous devez d'abord utiliser
    [coqc] pour compiler [Basics.v] et obtenir un fichier
    [Basics.vo]. Cela est semblable à obtenir un fichier .class (ou
    .o) à partir d'un fichier .java (ou .c). Il y a deux façons de
    faire cela :
    
    - Dans CoqIDE :
        
        Ouvrir [Basics.v]. Dans le menu "Compile", cliquer sur
        "Compile Buffer".
    
    - Dans un shell :

        [coqc Basics.v]

    Si quelque chose se passe mal (par ex. un message à propos
    d'identifieurs manquants), cela peut être à cause du "load path"
    de Coq qui n'est pas mis en place correctement. La commande [Print
    LoadPath.] peut vous aider pour dépêtrer les problèmes. *)
(* ################################################################# *)
(** * Preuve par induction (ou récurrence) *)

(** Dans le chapitre précédent, nous avons montré que [0] est un
    élément neutre à gauche pour [+] en utilisant un argument facile
    basé sur la simplification. Nous avons aussi observé que prouver
    que c'est aussi un élément neutre à _droite_ ...*)

Theorem plus_n_O_firsttry : forall n:nat,
  n = n + 0.

(** ... ne peut pas être fait de la même façon simple. Appliquer
    seulement [reflexivity] ne marche pas car le [n] dans [n + 0] est
    un entier arbitraire inconnu et ainsi le [match] dans la
    définition de [+] ne peut pas être simplifié. *)

Proof.
  intros n.
  simpl. (* Ne fait rien ! *)
Abort.

(** Raisonner par cas en utilisant [destruct n] ne nous mène pas
    beaucoup plus loin : La branche de l'analyse de cas où l'on
    suppose [n = 0] se prouve aisément, mais dans la branche où [n = S
    n'] pour un [n'] on se retrouve bloqué exactement de la même
    manière.  *)

Theorem plus_n_O_secondtry : forall n:nat,
  n = n + 0.
Proof.
  intros n. destruct n as [| n'].
  - (* n = 0 *)
    reflexivity. (* ok ça passe... *)
  - (* n = S n' *)
    simpl.       (* ... mais ici on est à nouveau bloqué *)
Abort.

(** On pourrait utiliser [destruct n'] pour aller un cran plus
    loin. Mais comme [n] peut être arbitrairement grand, si nous
    continuons ainsi cela ne s'arrête pas. *)

(** Pour prouver des faits intéressants à propos des nombres, listes
    ou d'autres types définis inductivement, on a besoin d'un principe
    de raisonnement plus puissant : l' _induction_ ou la
    _récurrence_. (Dans le domaine de la preuve sur ordinateur, par
    influence de l'anglais, l'habitude est de dire induction alors
    qu'en mathématique vous avez appris le raisonnement par
    récurrence.)

    Rappelons (des cours de mathématiques du lycée) le _principe de
    récurrence sur les entiers naturels_ : si [P(n)] est une
    proposition faisant intervenir un entier naturel [n] et que l'on
    veut montrer que [P] est vérifiée pour tout entier naturel [n], on
    peut raisonner de la façon suivante :
        - montrer que [P(0)]] est vérifiée,
        - montrer que, pour tout [n'], si [P(n')] est vérifiée, alors
          elle l'est pour [P(S n')],
        - conclure que [P(n)] est vérifiée pour tout [n]. *)
   
(** Dans Coq, les étapes sont les mêmes : on commence avec le but de
    prouver [P(n)] pour tout [n] et on le scinde (par l'application de
    la tactique [induction]) en deux sous-buts : l'un où l'on doit
    montrer [P(0)] et l'autre où l'on doit montrer que [P(n') ->
    P(S n')]. Voici comment cela marche avec le théorème que nous
    avons sous la main : *)

Theorem plus_n_O : forall n:nat, n = n + 0.
Proof.
  intros n. induction n as [| n' IHn'].
  - (* n = 0 *)    reflexivity.
  - (* n = S n' *) simpl. rewrite <- IHn'. reflexivity.  Qed.

(** Tout comme [destruct], la tactique [induction] prend une clause
    [as...] qui indique les noms des variables à introduire dans les
    sous-buts. Comme il y a deux sous-buts, la clause [as...] a deux
    parties séparées par un [|]. (Comme pour [destruct] et [intros],
    on peut omettre les clauses [as...] et Coq choisit les noms pour
    nous, ce qui n'est pas toujours une bonne idée.)

    Dans le premier sous-but, [n] est remplacé par [0]. Aucune
    nouvelle variable n'est introduite (et la première partie de la
    clause [as...] est vide), et le but devient [0 = 0 + 0] qui se
    montre par simplification.

    Dans le second sous-but, [n] est remplacé par [S n'] et l'hypothèse
    [n' + 0 = n'] est ajoutée au contexte avec le nom [IHn']
    (i.e. l'Hypothèse d'Induction sur [n']). Ces deux noms sont
    indiqués dans la seconde partie de la clause [as...]. Dans ce cas,
    le but devient [S n' = (S n') + 0] qui se simplifie en [S n' = S
    (n' + 0)] qui se déduit de [IHn']. *)

Theorem minus_diag : forall n,
  minus n n = 0.
Proof.
  intros n. induction n as [| n' IHn'].
  - (* n = 0 *)
    simpl. reflexivity.
  - (* n = S n' *)
    simpl. rewrite -> IHn'. reflexivity.  Qed.

(** L'utilisation de la tactique [intros] dans ces preuves est en fait
    redondant. Quand elle est appliqué à un but qui contient des
    variables quantifiées, la tactique [induction] va automatiquement
    les déplacer dans le contexte si besoin.*)

(** **** Exercice: **, recommandé (basic_induction)  *)
(** Prouvez les théorèmes suivants en utilisant l'induction. Vous
    devrez peut-être utiliser des résultats prouvés précédemment. *)

Theorem mult_0_r : forall n:nat,
  n * 0 = 0.
Proof.
  intros n. induction n as [| n' IHn'].
    - simpl. reflexivity.
    - simpl. rewrite -> IHn'. reflexivity.
Qed.

Theorem plus_n_Sm : forall n m : nat,
  S (n + m) = n + (S m).
Proof.
  intros n m. 
  induction n as [|n' IHn'].
    -simpl. reflexivity.
    -simpl. rewrite -> IHn'. reflexivity.
Qed.

Theorem plus_comm : forall n m : nat,
  n + m = m + n.
Proof.
  intros n m. 
  induction n as [|n' IHn'].
    -simpl. rewrite <- plus_n_O. reflexivity.
    -simpl. rewrite -> IHn'. rewrite <- plus_n_Sm. reflexivity.
Qed.

Theorem plus_assoc : forall n m p : nat,
  n + (m + p) = (n + m) + p.
Proof.
  intros n m p.
  induction n as [|n' IHn'].
    -simpl. reflexivity.
    -simpl. rewrite -> IHn'. reflexivity.
Qed. 

(** [] *)

(** **** Exercice: ** (double_plus)  *)
(** Considérez la fonction suivante qui double son argument : *)

Fixpoint double (n:nat) :=
  match n with
  | O => O
  | S n' => S (S (double n'))
  end.

(** Utilisez la tactique [induction] pour prouver le fait simple
    suivant à propos de [double] : *)

Lemma double_plus : forall n, double n = n + n .
Proof.
  intros n.
  induction n as [|n' IHn'].
    -simpl. reflexivity.
    -simpl. rewrite ->  IHn'. rewrite <- plus_n_Sm. reflexivity.
Qed.

(** [] *)

(** **** Exercice: **, optionnel (evenb_S)  *)
(** Un inconvénient de la définition de [evenb n] est l'appel
    récursif sur [n - 2]. Cela rend les preuves à propos de [evenb n]
    plus difficiles que s'il y avait une récurrence sur [n] car nous
    devons faire une hypothèse de récurrence sur [n - 2]. Le Lemme
    suivant donne une autre caractérisation de [evenb (S n)] plus
    adaptée à la récurrence : *)

Theorem evenb_S : forall n : nat,
  evenb (S n) = negb (evenb n).
Proof.
  intros n.
  induction n as [|n' IHn'].
    -simpl. reflexivity.
    -rewrite -> IHn'. simpl.  rewrite -> negb_involutive. reflexivity.
Qed.  

(** [] *)

(** **** Exercise: * (destruct_induction)  *)
(** Expliquez brièvement la différence entre les tactiques [destruct]
    et [induction] *)

(** 
La destruction permet de décomposer le but en fonction du constructeur du paramètre. Preuve par cas, ne fonctionne que si constructeur par récursif.
L'induction permet aussi de décomposer mais en posant aussi l'hypothèse de récurrence.
 *)

(** [] *)

(* ################################################################# *)
(** * Des preuves dans les preuves *)

(** En Coq, comme pour les mathématiques informelles, les grandes
    preuves sont souvent découpées en suites de théorèmes où les
    preuves des derniers théorèmes font références aux théorèmes
    précédents. Parfois, une preuve peut demander divers faits qui
    sont trop simples et/ou trop peu généraux qu'il n'y a pas
    d'intérêt à leur donner un nom. Dans de tels situations, il est
    utile de pouvoir énoncer et prouver le "sous-théorème" à l'endroit
    où on en a besoin. La tactique [assert] nous permet de faire
    cela. Par exemple, la preuve précédente du théorème [mult_0_plus]
    fait référence à un théorème montré auparavant nommé
    [plus_0_n]. Au lieu de procéder ainsi, on peut utiliser [assert]
    pour énoncer et prouver [plus_0_n] à la volée : *)

Theorem mult_0_plus' : forall n m : nat,
  (0 + n) * m = n * m.
Proof.
  intros n m.
  assert (H: 0 + n = n). { reflexivity. }
  rewrite -> H.
  reflexivity.  Qed.

(** La tactique [assert] introduit deux sous-buts. Le premier est
    l'assertion elle-même : en la préfixant avec [H:] on nomme
    l'assertion [H]. (On peut aussi nommer l'assertion avec une clause
    [as] comme on l'a fait auparavant avec [destruct] et [induction],
    i.e. [assert (0 + n = n) as H].) Remarquez que nous avons entouré
    la preuve de cette asertion avec des accolades. Cela à la fois
    pour la lisibilité et, quand Coq est utilisé interactivement, on
    voit plus facilement quand on a fini la sous-preuve. Le second but
    est le même que celui que l'on avait à l'endroit où l'on a utilisé
    [assert] sauf que nous avons maintenant l'hypothèse [H],
    c'est-à-dire [0 + n = n], dans le contexte. En résumé, [assert]
    génère un sous-but où l'on doit prouver le fait énoncé et un
    second où l'on peut utiliser le fait énoncé pour progresser dans
    la preuve que l'on veut faire.*)

(** Voici un autre exemple pour [assert]. On souhaite prouver que [(n + m) + (p
    + q) = (m + n) + (p + q)]. La seule différence entre les deux coté du signe
    [=] est que les deux opérandes [m] et [n] du premier [+] sont
    échangés. Ainsi, il semble que l'on devrait être capable d'utiliser la
    commutativité de l'addition ([plus_comm]) pour réécrire l'une des deux
    expressions en l'autre. Cependant, la tactique [rewrite] n'est pas très
    intelligente sur l' _endroit_ où elle applique la réécriture. Ici, il y a
    trois utilisation de [+] et, malheureusement, quand on écrit [rewrite ->
    plus_comm] cela n'a d'effet que sur le [+] le plus _extérieur_...  *)

Theorem plus_rearrange_firsttry : forall n m p q : nat,
  (n + m) + (p + q) = (m + n) + (p + q).
Proof.
  intros n m p q.
  (* On a simplement besoin de remplacer (n + m) en (m + n)... il
  semble que quelque chose comme plus_comm devrait le faire ! *)
  rewrite -> plus_comm.
  (* Cela ne marche pas... Coq réécrit le mauvais plus !*)
Abort.

(** Pour utiliser [plus_comm] à l'endroit où l'on en a besoin, on peut
    introduire un lemme local établissant que [n + m = m + n] (pour
    les [m] et [n] particuliers dont on parle ici), prouver ce lemme
    en utilisant [plus_comm], et ensuite l'utiliser pour la réécriture
    que l'on veut faire. *)

Theorem plus_rearrange : forall n m p q : nat,
  (n + m) + (p + q) = (m + n) + (p + q).
Proof.
  intros n m p q.
  assert (H: n + m = m + n).
  { rewrite -> plus_comm. reflexivity. }
  rewrite -> H. reflexivity.  Qed.

(* ################################################################# *)
(** * Preuves formelles et preuves informelles *)

(** "_Les preuves informelles sont les algorithmes, les preuves
    formelles sont le code_."*)

(** Qu'est-ce qui constitue une preuve réussie pour un mathématicien ?
    Cette question a occupé les philosophes pendant des millénaires,
    mais une réponse approximative peut être : Une preuve d'une
    proposition mathématique [P] est un texte écrit qui convainc
    graduellement le lecteur avec certitude que [P] est vraie -- un
    argument irréfutable de la vérité de [P]. Ainsi, une preuve est un
    acte de communication. *)

(** Ces actes de communication peuvent avoir différentes sortes de
    lecteurs. D'une part, le "lecteur" peut être un programme tel que
    Coq, dans ce cas la "croyance" qui est établie est que [P] peut
    être mécaniquement dérivée à partir d'un certain ensemble de
    règles logiques. La preuve est une recette qui guide le programme
    dans la vérification de ce fait. De telles recettes sont les
    preuves _formelles_. *)

(** D'autre part, le lecteur peut être un humain, dans un tel cas la
    preuve sera écrite dans une langue naturelle et sera
    nécessairement _informelle_. Ici, le critère de succès est moins
    clair. Une preuve "valide" est une preuve qui fera croire le
    lecteur à [P]. Mais la même preuve peut être lue par des lecteurs
    bien différents, certains d'entre eux seront convaincus par une
    façon particulière de conduire l'argumentation, alors que d'autres
    non. Pour des lecteurs particulièrement sourcilleux ou
    inexpérimentés, le seul moyen de les convaincre sera de détailler
    complètement l'argumentation. Pour d'autres lecteurs, plus
    familiers avec le domaine concerné, tout ces détails ne feront que
    les embrouiller et ils perdront le fil conducteur de la preuve. Ce
    que ces derniers voudront, c'est connaître les idées principales
    car il leur est facile de compléter les détails par eux-mêmes
    plutôt que de survoler leur présentation. Finalement, il n'y a pas
    de façon universelle d'écrire une preuve car il n'y a pas une
    seule façon d'écrire une preuve informelle qui garantisse de
    convaincre tous les lecteurs concevables. *)

(** Malgré tout, dans la pratique, les mathématiciens ont développé un
    riche ensemble de conventions et de langages pour écrire à propos
    d'objets mathématiques complexes qui -- au moins pour une certaine
    communauté -- rend la communication très sûre. Les conventions de
    cette forme épurée de communication donnent une façon sûre et
    standardisée pour s'assurer qu'une preuve est bonne ou
    mauvaise. *)

(** Comme nous utilisons Coq, nous travaillons principalement avec des
    preuves formelles. Cependant, cela ne signifie pas que l'on oublie
    complètement les preuves informelles. Les preuves formelles sont
    utiles de plein de manières, mais elles _ne sont pas_ un moyen très
    efficace pour communiquer des idées entre être humains. *)

(** Par exemple, voici une preuve que l'addition est associative : *)

Theorem plus_assoc' : forall n m p : nat,
  n + (m + p) = (n + m) + p.
Proof. intros n m p. induction n as [| n' IHn']. reflexivity.
  simpl. rewrite -> IHn'. reflexivity.  Qed.

(** Coq est parfaitement content de cela. Par contre, pour un être
    humain, cette preuve est difficile à comprendre. On utilise des
    commentaires et des tirets pour montrer la structure de la preuve
    un peu plus clairement... *)

Theorem plus_assoc'' : forall n m p : nat,
  n + (m + p) = (n + m) + p.
Proof.
  intros n m p. induction n as [| n' IHn'].
  - (* n = 0 *)
    reflexivity.
  - (* n = S n' *)
    simpl. rewrite -> IHn'. reflexivity.   Qed.

(** ... et si vous êtes familiers avec Coq, vous êtes peut être
    capable d'imaginer pas à pas l'état du contexte et le but à
    résoudre à chaque étape. Mais, si la preuve devient un peu plus
    compliquée, cela sera impossible.

    Un mathématicien (sourcilleux) pourrait écrire une preuve comme
    celle qui suit : *)

(** - _Théorème_: Pour tout [n], [m] et [p],

      n + (m + p) = (n + m) + p.

    _Preuve_: Par récurrence sur [n].

    - D'abord, supposons [n = 0].  On doit montrer

        0 + (m + p) = (0 + m) + p.

      Cela vient directement de la définition de [+].

    - Ensuite, supposons [n = S n'], où

        n' + (m + p) = (n' + m) + p.

      On doit montrer

        (S n') + (m + p) = ((S n') + m) + p.

      Par la définition de [+], on obtient l'égalité suivante

        S (n' + (m + p)) = S ((n' + m) + p),

      qui est établie immédiatement par l'hypothèse de récurrence. _Qed_. *)

(** La forme globale de cette preuve est essentiellement la même que
    la preuve en Coq et, bien sûr, cela n'est pas un accident : Coq a
    été conçu de telle façon que la tactique [induction] génère les
    mêmes sous-buts, et dans le même ordre, que les points qu'un
    mathématicien aurait écrit. Il y a tout de même des détails
    significativement différents : d'un certain point de vue, la
    preuve formelle est bien plus explicite (par ex. l'utilisation de
    [reflexivity]), mais, d'un autre point de vue, elle est moins
    explicite (en particulier, l'état de la preuve à un point donné
    est complètement implicite, alors que la preuve informelle
    rappelle plusieurs fois au lecteur où les choses en sont). *)

(** **** Exercice: **, recommandé (plus_comm_informal)  *)
(** Traduisez votre preuve pour [plus_comm] en une preuve informelle : *)
Theorem plus_comm' : forall n m : nat, n + m = m + n.
Proof.
  intros n m. 
  induction n as [|n' IHn'].
    -simpl. rewrite <- plus_n_O. reflexivity.
    -simpl. rewrite <- plus_n_Sm. rewrite -> IHn'. reflexivity.
Qed.

(** 
    Théorème: L'addition est commutative.

    Preuve : Par récurrence sur n,
        - D'abord, supposons [n = 0].  On doit montrer :
        0 + m = m + 0.
        Par la définition du +, on obtient m = m + 0.
        Avec la démonstration plus_n_O prouvée précédemment, on obtient m = m.

        - Ensuite, supposons que n = S n', où 
        n' + m = m + n' est l'hypothèse de récurrence.

        On doit montrer :
        S n' + m = m + S n'
        Par la définition du plus, on obtient S (n' + m) = m + S n'.
        Avec la démonstration plus_n_Sm prouvée précédemment, on obtient :
        S (n' + m) = S (m + n')
        qui est établie immédiatement par l'hypothèse de récurrence. _Qed_. 
*)

(** [] *)

Theorem beq_nat_n: forall n : nat, true = beq_nat n  n .
Proof.
  intros n.
  induction n as [| n' IHn'].
    -simpl. reflexivity.
    -simpl. rewrite -> IHn'. reflexivity.
Qed.


(** **** Exercice: **, optionnel (beq_nat_refl_informal)  *)
(** Écrivez une preuve informelle du théorème suivant, en utilisant la
    preuve informelle de [plus_assoc] comme modèle. Faites attention à
    ne pas paraphraser les tactiques Coq.

    Théorème: [true = beq_nat n n] pour tout [n].

    Preuve : Par récurrence sur n,
      - D'abord, supposons n = 0. 
      On doit montrer : (beq_nat 0  0) = true.
      Cela vient de la définition de beq_nat.

      - Ensuite, supposons que n = S n', où
      (beq_nat S n'  S n') = true.
      On retrouve notre hypothèse de récurrence. Qed.
[] *)

(* ################################################################# *)
(** * Exercices supplémentaires *)

(** **** Exercice: ***, recommandé (mult_comm)  *)
(** Utilisez [assert] pour vous aider à prouver le théorème
    suivant. Vous ne devez pas avoir à utiliser la récurrence sur
    [plus_swap]. *)

Theorem plus_swap : forall n m p : nat,
  n + (m + p) = m + (n + p).
Proof.
  intros n m p.
  rewrite -> plus_assoc. 
  rewrite -> plus_assoc. 
  assert (H: n + m = m + n).
  { rewrite -> plus_comm. reflexivity. }
  rewrite -> H. reflexivity.  Qed.

(** Maintenant prouvez la commutativité de la multiplication. Vous
    devrez probablement avoir besoin de définir et prouver un autre
    théorème séparément qui vous utiliserez ensuite dans la preuve de
    théorème. Vous trouverez sans doute que [plus_swap] est utile. *)


Theorem mult_aux : forall m n : nat, m * S n = m + m * n.
Proof.
  intros.
  induction m as [|m' IHm'].
  -simpl. reflexivity.
  -simpl. rewrite -> IHm'. rewrite <- plus_swap. reflexivity.
Qed.


Theorem mult_comm : forall m n : nat,
  m * n = n * m.
Proof.
  intros n m.
  induction n as [|n' IHn'].
    -simpl. rewrite -> mult_0_r. reflexivity.
    (*-assert( m * S n' = m + m * n'). 
     { induction m as [|m' IHm']. simpl. reflexivity. simpl. rewrite->IHm'. rewrite <- plus_swap. reflexivity. }
      + rewrite -> H. rewrite <- IHn'. reflexivity. *)
    -simpl. rewrite -> mult_aux. rewrite <- IHn'. reflexivity.
Qed. 

(** [] *)

(** **** Exercice: ***, optionnel *)
(** Pour chacun des théorèmes suivants, sur une feuille de papier,
    essayez d'anticiper si pour la preuve vous allez utiliser (a)
    seulement la simplification et la réécriture, (b) il va falloir
    aussi faire une analyse de cas ([destruct]), (c) un raisonnement
    par récurrence sera nécessaire. Écrivez vos prédictions et ensuite
    faites la preuve en Coq. *)

Check leb.

Theorem leb_refl : forall n:nat,
  true = leb n n.
Proof.
  intros n.
  induction n as [| n' IHn'].
  reflexivity.
  simpl.
  rewrite <- IHn'.
  reflexivity.
Qed.
  

Theorem zero_nbeq_S : forall n:nat,
  beq_nat 0 (S n) = false.
Proof.
  intros.
  simpl.
  reflexivity.
Qed.


Theorem andb_false_r : forall b : bool,
  andb b false = false.
Proof.
  intros. destruct b.
    - reflexivity.
    - reflexivity.
Qed.

Theorem plus_ble_compat_l : forall n m p : nat,
  leb n m = true -> leb (p + n) (p + m) = true.
Proof.
  intros.
  induction p as [|p' IHp'].
  - simpl. rewrite -> H. reflexivity.
  - simpl. rewrite -> IHp'. reflexivity.
Qed.


Theorem S_nbeq_0 : forall n:nat,
  beq_nat (S n) 0 = false.
Proof.
  intros.
  destruct n as [|n'].
  reflexivity.
  reflexivity.
Qed.

Theorem mult_1_l : forall n:nat, 1 * n = n.
Proof.
  intros.
  destruct n as [|n'].
    - reflexivity.
    - simpl. rewrite <- plus_n_O. reflexivity.
Qed.

Theorem all3_spec : forall b c : bool,
    orb
      (andb b c)
      (orb (negb b)
               (negb c))
  = true.
Proof.
  intros.
  destruct b.
    - destruct c.
      + reflexivity.
      + reflexivity.
    - destruct c.
      + reflexivity.
      + reflexivity.
Qed.
    

Theorem mult_plus_distr_r : forall n m p : nat,
  (n + m) * p = (n * p) + (m * p).
Proof.
  intros.
  induction p as [|p' IHp'].
    - simpl. rewrite -> mult_0_r. rewrite -> mult_0_r. rewrite -> mult_0_r. reflexivity.
    - Abort. (*rewrite  mult_assoc. rewrite <- IHp'.*)

Theorem mult_assoc : forall n m p : nat,
  n * (m * p) = (n * m) * p.
Proof.
  intros.
  destruct n as [|n'].
  simpl. reflexivity.
  simpl. rewrite -> mult_comm. 
  (* Remplir ici *) Admitted.
(** [] *)

(** **** Exercice: **, optionnel (beq_nat_refl)  *)
(** Prouvez le théorème suivant. (Mettre [true] à gauche du signe
    égal peut sembler maladroit, mais c'est ainsi que les théorème
    sont énoncés dans la bibliothèque standard de Coq, nous suivons
    donc cela. La réécriture fonctionne avec l'égalité dans les deux
    directions, donc nous n'aurons pas de problème pour utiliser ces
    théorèmes peu importe la façon dont ils sont énoncés.) *)

Theorem beq_nat_refl : forall n : nat,
  true = beq_nat n n.
Proof.
  (* Remplir ici *) Admitted.
(** [] *)

(** **** Exercice: **, optionnel (plus_swap')  *)
(** La tactique [replace] vous permet d'indiquer un sous-terme
    particulier à réécrire : [replace (t) with (u)] remplace dans le
    but (toutes les copies de) l'expression [t] par l'expression [u]
    et génère [t = u] comme but supplémentaire. Cela est souvent utile
    quand un [rewrite] global agit sur la mauvaise partie du but.

    Utilisez la tactique [replace] pour faire une preuve de
    [plus_swap'], tout comme [plus_swap], mais sans avoir besoin de
    [assert (n + m = m + n)]. *)

Theorem plus_swap' : forall n m p : nat,
  n + (m + p) = m + (n + p).
Proof.
  (* Remplir ici *) Admitted.
(** [] *)


