
(** Dans une première partie de ce chapitre, les éléments essentiels
    de la programmation en Coq sont introduits. Puis, dans une seconde
    partie, les premiers outils de construction des preuves de
    propriétés de programmes sont abordés. *)

(* ################################################################# *)
(** * Données et fonctions *)

(** Le système Coq est construit sur un nombre _extrêmement_ réduit de
    primitives. Plutôt que de fournir la palette usuelle de structure
    de données de base (les booléens, les entiers, les chaînes de
    caractères, etc.), le système Coq fournit des mécanismes puissants
    pour définir des structures de données. Cette approche permet de
    s'assurer que la correction du noyau de Coq pourra être vérifiée
    "à la main". *)

(** Naturellement, le système Coq est distribué avec un ensemble
    complet de bibliothèques qui fournissent les définitions (et les
    preuves des propriétés) de nombreuses structures de données comme
    les booléens, les nombres usuels, les listes, les tables de
    hachage, etc. *)

(* ================================================================= *)
(** ** Types énumérés *)

(* ----------------------------------------------------------------- *)
(** *** Jours de la semaine *)

(** Commençons par un exemple (très) simple. La déclaration ci-dessous
    indique à Coq que l'on définit de nouvelles valeurs qui forment un
    _type_. *)

(** Attention à ne pas oublier le point à la fin d'une déclaration. *)

Inductive day : Type :=
  | monday : day
  | tuesday : day
  | wednesday : day
  | thursday : day
  | friday : day
  | saturday : day
  | sunday : day.

(** Le type s'appelle [day] et ses valeurs sont [monday], [tuesday],
    etc.  À partir de la seconde ligne de la déclaration, on peut lire
    "[monday] est un [day]", "[tuesday] est un [day]", etc.

    Une fois le type [day] défini, on peut écrire des fonctions qui
    opèrent sur ses valeurs.  *)

Definition next_weekday (d:day) : day :=
  match d with
  | monday    => tuesday
  | tuesday   => wednesday
  | wednesday => thursday
  | thursday  => friday
  | friday    => monday
  | saturday  => monday
  | sunday    => monday
  end.

(** On peut remarquer que le type de l'argument et celui de la valeur
    de retour sont indiqués explicitement. Comme pour de nombreux
    langages typés (OCaml par exemple), Coq peut déduire le type
    d'une fonction (il fait une _inférence de types_). Les indications
    des types sont généralement présentes pour rendre la lecture du
    code plus facile. *)

(** On peut maintenant vérifier que la fonction calcule bien sur
    quelques exemples. Dans le système Coq, il y a trois façons de
    faire cela. La première est d'évaluer une expression contenant la
    fonction [next_weekday] avec la commande [Compute]. *)

Compute (next_weekday friday).
(* ==> monday : day *)

Compute (next_weekday (next_weekday saturday)).
(* ==> tuesday : day *)

(** %\noindent% Les réponses du système Coq que vous devez obtenir
    sont indiquées en commentaires. *)

(** %\vspace{0.5cm}% La seconde façon est d'indiquer ce que l'on _attend_ comme
    résultat sous la forme d'un exemple. *)

Example test_next_weekday:
  (next_weekday (next_weekday saturday)) = tuesday.

(** Cette déclaration fait deux choses : elle construit une assertion
    (le second jour ouvrable après [saturday] est [tuesday]) et elle
    donne un nom à l'assertion qui pourra être utilisé pour y faire
    référence plus tard. Maintenant que l'on a une assertion, on peut
    demander à Coq de la vérifier, comme cela : *)

Proof. simpl. reflexivity.  Qed.

(** Pour le moment, les détails ne sont pas importants. Le texte ci-dessus se
    lit de façon informelle comme "L'assertion que l'on vient de faire est
    prouvée en observant que les deux termes de chaque coté de signe égal
    s'évalue en une même valeur après quelques simplifications."

    La troisième façon est de demander à Coq d' _extraire_ de notre [Definition]
    un programme dans un langage de programmation plus conventionnel (OCaml,
    Scheme ou Haskell) dotés d'un compilateur performant.  Cette fonctionnalité
    est très intéressante car elle permet de transformer des programmes prouvés
    comme corrects en un code machine efficace. Bien sûr, il faire confiance à
    la correction du mécanisme d'extraction et à celle des compilateurs. En
    fait, la possibilité d'extraire un programme à partir d'une preuve est l'une
    des principales raisons pour lesquelles Coq a été développé. Nous
    reviendrons sur ce sujet au fil du cours. *)

(* ================================================================= *)
(** ** Les booléens *)

(** Les booléens sont définis de façon similaire à ce que nous venons
    de faire. Le type [bool] est défini avec deux valeurs [true] et
    [false]. *)

Inductive bool : Type :=
  | true : bool
  | false : bool.

(** Nous définissons ici nos propres booléens. Bien évidemment, Coq
    fournit de façon standard une définition des booléens ainsi que
    toutes les fonctions utiles et les preuves de leurs
    propriétés. Vous pouvez voir cela dans la documentation de la
    bibliothèque standard à la section [Coq.Init.Datatypes]. Pour le
    cours, quand cela sera possible, les noms des définitions et des
    théorèmes (les propriétés) coïncideront avec ceux de la
    bibliothèque standard.

    Les fonctions sur les booléens sont définies de façon semblable à
    celle vue précédemment : *)

Definition negb (b:bool) : bool :=
  match b with
  | true => false
  | false => true
  end.

Definition andb (b1:bool) (b2:bool) : bool :=
  match b1 with
  | true => b2
  | false => false
  end.

Definition orb (b1:bool) (b2:bool) : bool :=
  match b1 with
  | true => true
  | false => b2
  end.

(** Les deux dernières fonctions illustrent la syntaxe de Coq pour les
    fonctions à plusieurs arguments. Leur utilisation est, elle,
    illustrée ci-dessous pour les "tests unitaires" qui constituent une
    spécification complète de la fonction [orb]. On donne, en fait,
    une table de vérité. *)

Example test_orb1:  (orb true false) = true.
Proof. simpl. reflexivity.  Qed.
Example test_orb2:  (orb false false) = false.
Proof. simpl. reflexivity.  Qed.
Example test_orb3:  (orb false true)  = true.
Proof. simpl. reflexivity.  Qed.
Example test_orb4:  (orb true  true)  = true.
Proof. simpl. reflexivity.  Qed.

(** Coq fournit des facilités syntaxiques assez souples. On peut
    introduire facilement des notations usuelles pour les fonctions
    que l'on définit. La commande [Notation] introduit une nouvelle
    notation symbolique pour des définitions existantes. *)

Notation "x && y" := (andb x y).
Notation "x || y" := (orb x y).

Example test_orb5:  false || false || true = true.
Proof. simpl. reflexivity. Qed.

(** _Remarque sur les exercices_: La commande [Admitted] permet
    d'indiquer que l'on admet (temporairement) une propriété. Cela
    permet d'avoir des preuves incomplètes. Pour les exercices, cela
    permet de vous indiquer les endroits où vous devez remplir le
    texte manquant. *)

(** **** Exercice: * (nandb)  *)

(** Supprimez [Admitted] et donnez la définition de la fonction
    [nanb]. Assurez-vous ensuite que les assertions [Example]
    suivantes peuvent être vérifiées par Coq. Pour faire les preuves
    des assertions prenez exemple sur le modèle des preuves des
    assertions pour la fonction [orb]. La fonction [nanb] doit
    renvoyer [true] si l'un ou les deux arguments sont [false]. *) 

Definition nandb (b1:bool) (b2:bool) : bool :=
  (** Remplacez cette ligne par ":= _votre définition_" *)
  match b1 with
    | true => negb b2
    | false => true 
  end.

Example test_nandb1:               (nandb true false) = true.
Proof. simpl. reflexivity. Qed.

Example test_nandb2:               (nandb false false) = true.
Proof. simpl. reflexivity. Qed.

Example test_nandb3:               (nandb false true) = true.
Proof. simpl. reflexivity. Qed.

Example test_nandb4:               (nandb true true) = false.
Proof. simpl. reflexivity. Qed.

(** [] *)

(** **** Exercice: * (andb3)  *)
(** Faites comme précédemment pour la fonction [andb3]
    ci-dessous. Cette fonction doit renvoyer [true] quand tous ses
    arguments sont [true] et [false] sinon. *)

Definition andb3 (b1:bool) (b2:bool) (b3:bool) : bool :=
  match b1 with
    | true => andb b2 b3 
    | false => false
  end.

Example test_andb31:                 (andb3 true true true) = true.
Proof. simpl. reflexivity. Qed.

Example test_andb32:                 (andb3 false true true) = false.
Proof. simpl. reflexivity. Qed.

Example test_andb33:                 (andb3 true false true) = false.
Proof. simpl. reflexivity. Qed.

Example test_andb34:                 (andb3 true true false) = false.
Proof. simpl. reflexivity. Qed.

(** [] *)

(* ================================================================= *)
(** ** Le type des fonctions *)

(** Toute expression en Coq a un type qui décrit quelle sorte d'objets
    elle calcule. La commande [Check] demande à Coq d'afficher le type
    d'une expression. *)

Check true.
(* ===> true : bool *)
Check (negb true).
(* ===> negb true : bool *)

(** Les fonctions comme [negb] sont aussi des valeurs, tout comme
    [true] ou [false]. Leur types sont appelés des _types fonctions_
    et ils sont écrits avec des flèches. *)

Check negb.
(* ===> negb : bool -> bool *)

(** Le type de [negb] s'écrit [bool -> bool] et se prononce "[bool]
    flèche [bool]". Cela se lit "Étant donné un argument de type
    [bool], cette fonction renvoie une valeur de type [bool]".

    De la même façon, le type de [andb] s'écrit [bool -> bool -> bool]
    et se lit "Étant donnés deux arguments de type [bool], cette
    fonction produit une valeur de type [bool]".*)

(* ================================================================= *)
(** ** Types composés *)

(** Les types que nous avons défini jusqu'à présent sont des exemples
    de "types énumérés" : leur définition énumère explicitement les
    valeurs de ces types. Chaque valeur est un constructeur
    simple. Voici ci-après une définition de type plus complexe où
    l'un des constructeurs prend un argument : *)

Inductive rgb : Type :=
  | red : rgb
  | green : rgb
  | blue : rgb.

Inductive color : Type :=
  | black : color
  | white : color
  | primary : rgb -> color.

(** Regardons cela de façon plus détaillée.

    Chaque type défini inductivement ([day], [bool], [rgb], [color],
    etc.) contient des expressions construites avec des
    _constructeurs_ comme [red], [primary], [true], [false], [monday],
    etc. Les définitions de [rgb] et [color] disent comment les
    expressions des types [rgb] et [color] peuvent être construites :

    - [red], [green] and [blue] sont les constructeurs de [rgb],
    - [black], [white] et [primary] sont les constructeurs de [color],
    - l'expression [red] appartient au type [rgb], de même pour les
    - expressions [green] et [blue],
    - les expressions [black] et [white] appartiennent au type
      [color],
    - si [p] est une expression du type [rgb] alors [primary p]
      (i.e. le constructeur [primary] appliqué à l'argument [p]) est
      une expression du type [color] et,
    - les expressions formées de ces façons sont les _seules_ qui sont
      des types [rgb] et [color]. *)

(** On peut définir des fonctions sur les valeurs du type [color] en
    utilisant le filtrage comme on l'a fait pour les valeurs des types
    [day] et [bool]. *)

Definition monochrome (c : color) : bool :=
  match c with
  | black => true
  | white => true
  | primary p => false
  end.

(** Comme le constructeur [primary] prend un argument, un filtrage sur
    [primary] doit contenir soit une variable (comme ci-dessus) ou une
    constante du type approprié (comme ci-dessous). *)

Definition isred (c : color) : bool :=
  match c with
  | black => false
  | white => false
  | primary red => true
  | primary _ => false
  end.

(** Le motif [primary _] est un raccourci pour "[primary] appliqué à
    tout constructeur de [rgb]" (sauf [red] car il a déjà été filtré
    avant). Le motif joker [_] a le même effet que la variable muette
    [p] dans la définition de [monochrome]. *)

(* ================================================================= *)
(** ** Les modules *)

(** Coq fournit un _système de modules_ pour aider à l'organisation de
    grands développements. Dans ce cours, nous n'aurons pas besoin de
    la plupart de ses fonctionnalités, mais une nous est utile : Si
    nous encadrons une collection de déclarations entre les marqueurs
    [Module X] et [End X] alors, dans la suite du fichier, après [End
    X], ces définitions sont référencées par des noms comme [X.foo] au
    lieu de seulement [foo]. Cette fonctionnalité est utilisée pour
    introduire la définition du type [nat] dans un module interne pour
    qu'il n'interfère pas avec le type de la bibliothèque standard
    (que nous utilisons par la suite car il vient avec des notations
    spéciales qui sont bien pratiques). *)

Module NatPlayground.

(* ================================================================= *)
(** ** Les entiers naturels *)

(** Une façon très intéressante et utile de définir un type est que
    ses constructeurs prennent des arguments de ce type. Ainsi, les
    expressions qui décrivent les valeurs de ce type sont
    _inductives_.

    Par exemple, on peut définir (une représentation des) les entiers
    naturels comme suit : *)

Inductive nat : Type :=
  | O : nat
  | S : nat -> nat.

(** Les clauses de cette définition peuvent se lire :

    - [O] est un entier naturel (remarquez qu'il s'agit de la lettre
      "[O]" pas du chiffre "[0]").
    - [S] peut être mis devant un entier naturel pour construire un
      nouvel entier naturel. Si [n] est un entier naturel alors [S n]
      en est un aussi. *)

(** À nouveau, regardons cela plus en détail. La définition de [nat]
    dit comment les expressions du type [nat] sont construites :

    - [O] et [S] sont des constructeurs,
    - l'expression [O] est du type [nat],
    - si [n] est une expressions du type [nat] alors [S n] est aussi
      une expression du type [nat], et
    - les expressions formées de ces deux manières sont les seules qui
      sont du type [nat]. *)

(** Ce sont les mêmes qui s'appliquent que pour les définitions de [day],
    [bool], [color], etc.
    
    Ces règles sont précisément la force des déclarations [Inductive]. Elles
    impliquent que les expressions [O], [S O], [S (S O)], [S (S (S O))] et ainsi
    de suite sont du type [nat]. Alors que les expressions comme [true], [andb
    true false], [S (S false)] et [O (O (O s))] ne le sont pas.
    
    Un point essentiel est que ce que nous venons de faire est simplement de
    définir une _représentation_ des entiers naturels : une façon de les
    écrire. Les noms [O] et [S] sont arbitraires. Et, pour le moment, ils n'ont
    pas de signification particulière, il s'agit juste des deux marqueurs
    différents pour écrire des entiers naturels, associés à une règle qui dit
    que tout entier naturel s'écrit comme une suite de marqueurs [S] suivi d'un
    marqueur [O]. Si on le préfère, le type [nat] peut s'écrire de la façon
    suivante : *)

Inductive nat' : Type :=
  | stop : nat'
  | tick : nat' -> nat'.

(** L' _interprétation_ de ces marqueurs vient de la façon dont on les
    utilise pour calculer. *)

(** On peut faire cela en écrivant des fonctions dont le filtrage
    coïncide avec la représentation des entiers naturels tout comme
    nous l'avons fait pour les booléens et les jours. Par exemple,
    voici la fonction prédécesseur :*)

Definition pred (n : nat) : nat :=
  match n with
    | O => O
    | S n' => n'
  end.

(** La seconde branche se lit : "si [n] est de la forme [S n'] pour un
    [n'] quelconque alors renvoyer [n']." *) 

End NatPlayground.

(** Parce que les entiers naturels sont une forme de données tellement
    répandue, Coq fournit un petit nombre de notations prédéfinies pour
    les utiliser. Les chiffres décimaux usuels peuvent être utilisés à
    la place de la notation unaire définie par les constructeurs [O]
    et [S]. Par défaut, Coq affiche les entiers naturels avec les
    chiffres décimaux : *)

Check (S (S (S (S O)))).
  (* ===> 4 : nat *)

Definition minustwo (n : nat) : nat :=
  match n with
    | O => O
    | S O => O
    | S (S n') => n'
  end.

Compute (minustwo 4).
  (* ===> 2 : nat *)

(** Remarquez que le constructeur [S] a comme type [nat -> nat] tout
    comme les fonctions [pred] et [minustwo] : *)

Check S.
Check pred.
Check minustwo.

(** Toutes ces expressions peuvent être appliquées à un entier naturel pour
    obtenir un nouvel entier naturel. Cependant, il y a une différence
    fondamentale entre la première et les deux suivantes : les fonctions comme
    [pred] et [minustwo] ont des _règles de calcul_, la définition de [pred] dit
    que [pred 2] peut être simplifiée en [1]. Tandis que la définition de [S]
    n'est pas associée à un tel comportement. Malgré cela, on voit le
    constructeur [S] comme une fonction car il peut être appliqué à un argument,
    mais il ne fait pas de calcul. Il s'agit simplement d'un moyen d'écrire les
    entiers naturels. Vous pouvez faire l'analogie avec l'écriture décimale d'un
    nombre, écrire [1] n'est pas un calcul, c'est une donnée. Et, quand on écrit
    [111] pour désigner l'entier naturel cent onze, on utilise trois fois [1]
    pour écrire une représentation de ce nombre.

     Pour la plupart des fonctions sur les entiers naturels, la seule
     utilisation du filtrage n'est pas suffisante : on doit aussi utiliser la
     récursion. Par exemple, pour vérifier si [n] est pair, on peut
     récursivement vérifier que [n-2] est pair. Pour écrire des fonctions
     récursives, on utilise le mot-clef [Fixpoint]. *)

Fixpoint evenb (n:nat) : bool :=
  match n with
  | O        => true
  | S O      => false
  | S (S n') => evenb n'
  end.

(** On peut définir la fonction [oddb] par une déclaration [Fixpoint]
    semblable, mais voici une définition plus simple :*)

Definition oddb (n:nat) : bool   :=   negb (evenb n).

Example test_oddb1:    oddb 1 = true.
Proof. simpl. reflexivity.  Qed.
Example test_oddb2:    oddb 4 = false.
Proof. simpl. reflexivity.  Qed.

(** (Si vous exécutez ces preuves pas à pas, vous pouvez remarquer que [simpl]
    n'a, en fait, pas d'effet sur le but -- tout le travail est fait par
    [reflexivity]. On va voir cela plus en détail dans la suite.)
*)

(** Bien sûr, des fonctions récursives à plusieurs arguments peuvent
    aussi être définies. *)

Module NatPlayground2.

Fixpoint plus (n : nat) (m : nat) : nat :=
  match n with
    | O => m
    | S n' => S (plus n' m)
  end.

(** Et, additionner trois à deux donne le résultat attendu. *)

Compute (plus 3 2).

(** La simplification qu'effectue Coq pour obtenir ce résultat peut être visualisée comme suit : *)

(** <<  [plus (S (S (S O))) (S (S O))]
==> [S (plus (S (S O)) (S (S O)))]
      par la seconde clause du filtrage
==> [S (S (plus (S O) (S (S O))))]
      par la seconde clause du filtrage
==> [S (S (S (plus O (S (S O)))))]
      par la seconde clause du filtrage
==> [S (S (S (S (S O))))]
      par la première clause du filtrage
>> *)

(** Une facilité de notation : si deux arguments ou plus ont le même type, ils
    peuvent être écrits ensembles dans la déclaration d'une fonction. Dans la
    définition suivante, [(n m : nat)] signifie la même chose si nous avions
    écrit [(n : nat) (m : nat)]. *)

Fixpoint mult (n m : nat) : nat :=
  match n with
    | O => O
    | S n' => plus m (mult n' m)
  end.

Example test_mult1: (mult 3 3) = 9.
Proof. simpl. reflexivity.  Qed.

(** On peut filtrer des expressions en une seule fois en mettant une
    virgule entre les deux expressions :*)

Fixpoint minus (n m:nat) : nat :=
  match (n, m) with
  | (O   , _)    => O
  | (S _ , O)    => n
  | (S n', S m') => minus n' m'
  end.

(** À nouveau, le "_" dans la première ligne du filtrage est un _motif
    joker_. Écrire [_] dans un motif est la même chose qu'écrire une
    variable qui n'est pas utilisée dans la partie droite du
    filtrage. Cela permet d'éviter de devoir inventer un nom de
    variable.*)

End NatPlayground2.

Fixpoint exp (base power : nat) : nat :=
  match power with
    | O => S O
    | S p => mult base (exp base p)
  end.

(** **** Exercice: * (factorielle)  *)
(** On rappelle ci-dessous la définition de la fonction factorielle :

       factorial(0)  =  1
       factorial(n)  =  n * factorial(n-1)     (si n>0)

     Traduisez cela dans Coq.*)

Fixpoint factorial (n:nat) : nat :=
    match n with
      | O =>1
      | S n' => mult n (factorial n')
    end.

Example test_factorial1:          (factorial 3) = 6.
Proof. simpl. reflexivity. Qed.

Example test_factorial2:          (factorial 5) = (mult 10 12).
Proof. simpl. reflexivity. Qed.


(** [] *)

(** On peut rendre les expressions numériques un peu plus faciles à
    lire en introduisant des _notations_ pour l'addition, la
    multiplication et la soustraction. *)

Notation "x + y" := (plus x y)
                       (at level 50, left associativity)
                       : nat_scope.
Notation "x - y" := (minus x y)
                       (at level 50, left associativity)
                       : nat_scope.
Notation "x * y" := (mult x y)
                       (at level 40, left associativity)
                       : nat_scope.

Check ((0 + 1) + 1).

(** Les annotations [level], [associativity] et [nat_scope] contrôlent
    comment ces notations sont traitées par le parseur de Coq. Les
    détails ne sont pas importants pour notre propos (plus
    d'information peut être trouvée dans le manuel de Coq).

    Remarquons que les notations ne changent pas les définitions que
    nous avons faites : elles sont seulement des instructions au
    parseur de Coq pour accepter [x+y] à la place de [plus x y] et,
    réciproquement, l'affichage de Coq affichera [plus x y] comme [x +
    y]. *)

(** Quand nous avons dit que Coq est livré avec presqu'aucune
    fonctionnalité prédéfinie, il faut bien comprendre que c'est
    effectivement le cas : même le test d'égalité entre deux nombres
    naturels est une opération définie par l'utilisateur ! On définit
    maintenant une fonction [beq_nat], qui teste l'égalité des entiers
    naturels en produisant un booléen. Remarquez l'utilisation des
    filtrages emboîtés (on aurait aussi pu utiliser un filtrage
    simultané comme cela a été fait pour [minus]). *)

Fixpoint beq_nat (n m : nat) : bool :=
  match n with
  | O => match m with
         | O => true
         | S m' => false
         end
  | S n' => match m with
            | O => false
            | S m' => beq_nat n' m'
            end
  end.

(** La fonction [leb] teste si son premier argument est inférieur ou
    égal à son premier argument en renvoyant un booléen. *)

Fixpoint leb (n m : nat) : bool :=
  match n with
  | O => true
  | S n' =>
      match m with
      | O => false
      | S m' => leb n' m'
      end
  end.

Example test_leb1:             (leb 2 2) = true.
Proof. simpl. reflexivity.  Qed.
Example test_leb2:             (leb 2 4) = true.
Proof. simpl. reflexivity.  Qed.
Example test_leb3:             (leb 4 2) = false.
Proof. simpl. reflexivity.  Qed.

(** **** Exercice: * (blt_nat)  *)
(** La fonction [blt_nat] teste si un entier naturel est plus petit
    qu'un autre et renvoie un booléen. Au lieu d'écrire un nouveau
    [Fixpoint] pour cette fonction, définissez-la en termes de la
    fonction définie précédemment. *)

Definition blt_nat (n m : nat) : bool :=
    (leb n m) && negb (beq_nat n m).


Example test_blt_nat1:             (blt_nat 2 2) = false.
Proof. simpl. reflexivity. Qed.

Example test_blt_nat2:             (blt_nat 2 4) = true.
Proof. simpl. reflexivity. Qed.

Example test_blt_nat3:             (blt_nat 4 2) = false.
Proof. simpl. reflexivity. Qed.

(** [] *)

(* ################################################################# *)
(** * Preuve par simplification *)

(** Maintenant que nous avons défini quelques structures de données
    et fonctions, il est temps d'établir et de prouver quelques
    propriétés de leur comportement. En fait, nous avons déjà commencé
    à faire cela : chaque [Example] dans les paragraphes précédents
    est une assertion précise à propos du comportement d'une certaine
    fonction appliquée à certains arguments. Les preuves de ces
    assertions ont été toujours les mêmes : utiliser [simpl] pour
    simplifier les deux membres de l'équation et ensuite utiliser
    [reflexivity] pout vérifier que les deux membres de l'équation
    contiennent des valeurs identiques.

    Le même type de "preuve par simplification" peut être utiliser
    pour prouver des propriétés plus intéressantes. Par exemple, le
    fait que [0] est un "élément neutre à gauche" pour [+] peut être
    prouvé en observant simplement que [0 + n] se réduit à [n], peu
    importe [n], c'est un fait qui peut être lu directement de la
    définition de [plus].*)

Theorem plus_O_n : forall n : nat, 0 + n = n.
Proof.
  intros n. simpl. reflexivity.  Qed.

(** C'est le bon moment pour indiquer que la tactique [reflexivity]
    est un peu plus puissante que suggéré jusqu'à présent. Dans les
    exemples que l'on a vu, les appels à [simpl] n'étaient en fait pas
    nécessaires, car [reflexivity] peut effectuer quelques
    simplifications automatiquement quand elle vérifie les deux
    membres d'une équations sont égaux : [simpl] a simplement été
    ajouté afin de voir l'étape intermédiaire -- après la
    simplification et avant de finir la preuve. Voici une preuve plus
    courte du théorème précédent : *)

Theorem plus_O_n' : forall n : nat, 0 + n = n.
Proof.
  intros n. reflexivity. Qed.

(** Cependant, pour la suite, il est utile de savoir que [reflexivity]
    fait en quelque sorte _plus_ de simplifications qu'en fait [simpl]
    -- par exemple, elle essaie de "déplier" les termes des
    définitions en les remplaçant par les membres droits de ces
    définitions. La raison de cette différence est que, si
    [reflexivity] réussi, le but est prouvé et l'on n'a pas besoin de
    regarder les expressions développées que [reflexivity] a créé par tous
    les dépliements et les simplifications. Par contraste, [simpl] est
    utilisé dans les situations où l'on voudrait lire et comprendre le
    nouveau but qu'elle crée. Dans ce cas, nous ne voulons pas que les
    définitions soient développées à l'aveugle et obtenir un but dans
    un état "désordonné".

    La forme du théorème que l'on vient d'établir et sa preuve sont à
    peu près les mêmes que les exemples simples que nous avons vu
    précédemment, il y a tout de même quelques différences.

    La première, on a utilisé le mot-clef [Theorem] au lieu de
    [Example]. Cette différence est principalement une question de
    style : les mots-clefs [Example] et [Theorem] (ainsi que quelques
    autres, comme [Lemma], [Fact] et [Remark]) signifient quasiment la
    même chose en Coq.

    La seconde différence est que nous avons ajouté le quantificateur
    [forall n:nat], ainsi notre théorème concerne _tous_ les entiers
    naturels [n]. De façon informelle, pour prouver un théorème de
    cette forme, on commence généralement par dire "Supposons que [n]
    soit un entier naturel..." De façon formelle, cette supposisiton
    est faite dans la preuve par [intros n], qui déplace [n] du
    quantificateur du but dans le _contexte_ des hypothèses courantes.

     Les mots-clefs [intros], [simpl] et [reflexivity] sont des
     exemples de _tactiques_. Une tactique est une commande que l'on
     utilise entre les mots-clefs [Proof] et [Qed] pour guider le
     processus de vérification d'une assertion que l'on fait. Nous
     allons voir d'autres tactiques dans la suite et plus encore dans
     les autres chapitres. *)

(** D'autres théorèmes similaires peuvent être prouvés en utilisant la
    même structure. *)

Theorem plus_1_l : forall n:nat, 1 + n = S n.
Proof.
  intros n. reflexivity.  Qed.

Theorem mult_0_l : forall n:nat, 0 * n = 0.
Proof.
  intros n. reflexivity.  Qed.

(** Le suffixe [_l] dans les noms de ces théorème doit se lire "à
gauche". *)

(** En effectuant pas à pas ces preuves, il est intéressant d'observer
    comment le contexte et le but se modifient. Vous pouvez ajouter un
    appel à [simpl] avant [reflexivity] pour voir les simplifications
    que Coq effectue sur les termes avant de vérifier qu'ils sont
    égaux.*)

(* ################################################################# *)
(** * Preuve par réécriture *)

(** Le théorème ci-dessous est un peu plus intéressant que ceux que
    l'on a vu jusqu'à maintenant :*)

Theorem plus_id_example : forall n m:nat,
  n = m ->
  n + n = m + m.

(** Au lieu de faire une assertion universelle à propos de tous les
    nombres [n] et [m], il considère une propriété plus spécialisée
    qui est valide seulement quand [n = m]. Le symbole de la flèche se
    prononce "implique".

    Comme auparavant, nous devons être capable de raisonner en
    supposant que nous nous donnons de tels entiers naturels [n] et
    [m]. Nous devons aussi introduire l'hypothèse [n = m]. La tactique
    [intros] sert à déplacer les trois hypothèses du but dans les
    hypothèses du contexte courant.

     Comme [n] et [m] sont des entiers naturels arbitraires, on ne
     peut pas utiliser la simplification pour prouver ce théorème. À
     la place, on le prouve en observant que, si l'on suppose [n = m],
     alors on peut remplacer [n] par [m] dans le but et obtenir une
     égalité avec les mêmes expressions des deux cotés. La tactique
     qui dit à Coq d'effectuer ce remplacement s'appelle [rewrite]. *)

Proof.
  (* déplace les variables quantifiées dans le contexte *)
  intros n m.
  (* déplace l'hypothèse dans le contexte *)
  intros H.
  (* réécrit le but en utilisant l'hypothèse *)
  rewrite -> H.
  reflexivity.  Qed.

(** La première ligne de la preuve déplace les variables quantifiées
    universellement [n] et [m] dans le contexte. La deuxième ligne
    déplace l'hypothèse [n = m] dans le contexte et lui donne le nom
    [H]. La troisième ligne dit à Coq de réécrire le but courant ([n
    +n = m + m]) en remplaçant le coté gauche de l'égalité de
    l'hypothèse [H] avec le coté droit.

    Le symbole de la flèche dans le [rewrite] n'a rien à voir avec
    l'implication : il dit à Coq d'applique la réécriture de la gauche
    vers la droite. Pour réécrire de la droite vers la gauche, vous
    pouvez utiliser [rewrite <-]. Essayez cela dans la preuve
    précédente pour voir la différence que cela produit. *)

(** **** Exercice: * (plus_id_exercise)  *)
(** Enlevez "[Admitted.]" et remplissez la preuve. *)

Theorem plus_id_exercise : forall n m o : nat,
  n = m -> m = o -> n + m = m + o.
Proof.
  intros n m o.
  intros H.
  intros H0.
  rewrite -> H.
  rewrite -> H0.
  reflexivity.
Qed.
(** [] *)

(** La commande [Admitted] dit à Coq que l'on ne veut pas faire la
    preuve du théorème et l'accepter tel qu'il est donné. Cela peut
    être utile quand on développe de longues preuves, ainsi on peut
    établir des lemmes intermédiaires que l'on pense utiliser pour
    construire un argument plus grand, [Admitted] est utilisé pour
    accepter ces lemmes comme sûrs pour un moment, afin de continuer à
    travailler sur l'argument principal jusqu'à ce que l'on soit
    certain que cela fait sens : alors on revient en arrière et l'on
    remplit les preuves manquantes. Attention toutefois, car chaque
    fois que l'on utilise [Admitted] on laisse la porte ouverte à une
    contradiction qui peut entrer dans le monde rigoureux et
    formellement vérifié de Coq ! *)

(** On peut aussi utiliser la tactique [rewrite] avec un théorème
    prouvé précédemment au lieu d'une hypothèse du contexte. Si
    l'expression du théorème prouvé auparavant utilise des variables
    quantifiées, comme dans l'exemple ci-dessous, Coq essaie des les
    instancier en filtrant avec le but courant. *)

Theorem mult_0_plus : forall n m : nat,
  (0 + n) * m = n * m.
Proof.
  intros n m.
  rewrite -> plus_O_n.
  reflexivity.  Qed.

(** **** Exercice: ** (mult_S_1)  *)
Theorem mult_S_1 : forall n m : nat,
  m = S n ->
  m * (1 + n) = m * m.
Proof.
  intros n m.
  intros H0.
  rewrite -> plus_1_l.
  rewrite -> H0.
  reflexivity.
Qed.

  (* Remarque : cette preuve peut en fait être faite sans utiliser
  [rewrite], pour l'exercice utilisez [rewrite]. *)
(** [] *)

(* ################################################################# *)
(** * Preuve par analyse de cas *)

(** Bien évidemment, tout ne peut pas se prouver par des calculs et
    des réécritures simples. En général, des valeurs (des nombres
    quelconques, des booléens, des listes, etc.) inconnues,
    hypothétiques peuvent bloquer la simplification. Par exemple, si
    nous essayons de prouver le théorème ci-dessous en utilisant la
    tactique [simpl] comme avant, on reste coincé. On utilise alors la
    commande [Abort] pour abandonner la preuve un moment. *)

Theorem plus_1_neq_0_firsttry : forall n : nat,
  beq_nat (n + 1) 0 = false.
Proof.
  intros n.
  simpl.  (* ne fait rien *)
Abort.

(** La raison de ce bloquage est que les deux définitions [beq_nat] et
    [+] commencent par effectuer un filtrage sur leur premier
    argument. Mais ici, le premier argument de [+] est l'entier
    inconnu [n] et l'argument de [beq_nat] est l'expression composée
    [n + 1], aucune ne peut être simplifiée.

    Pour progresser, on a besoin de considérer les différentes formes
    possibles de [n] séparément. Si [n = 0], alors on peut
    calculer le résultat final de [beq_nat (n + 1) 0] et vérifier que
    c'est en fait [false]. Et, si [n = S n'] pour un [n'] quelconque,
    alors, même si l'on connaît pas exactement quel nombre est produit
    par [n + 1], on peut calculer qu'au moins il va commencer par un
    [S], et cela est suffisant pour calculer que [beq_nat (n + 1) 0]
    vaut [false] à nouveau.

    La tactique qui dit à Coq de considérer, séparément, les cas où [n
    = 0] et [n = S n'] s'appelle [destruct]. *)

Theorem plus_1_neq_0 : forall n : nat,
  beq_nat (n + 1) 0 = false.
Proof.
  intros n. destruct n as [| n'].
  - reflexivity.
  - reflexivity.   Qed.

(** L'application de [destruct] génère _deux_ sous-buts, que l'on doit
    maintenant prouver afin que Coq accepte le théorème. L'annotation "[as [|
    n']]" est appelée un _motif d'introduction_. Il dit à Coq quel noms de
    variable à introduire dans chaque sous-but. En général, ce que l'on met
    entre les crochets est une _liste de listes_ de noms, séparés par [|]. Dans
    notre cas, le premier composant est vide car le constructeur [O] n'a pas
    d'argument. Le second composant donne un seul nom, [n'] car le constructeur
    [S] a un seul argument.

    Les signes [-] des deuxièmes et troisièmes lignes sont des _tirets_ qui
    marquent la part de la preuve qui correspond à chaque sous-but généré. Le
    script de preuve qui vient après un tiret est la preuve entière du
    sous-but. Dans cet exemple, chaque sous-but est facilement prouvé par une
    seule utilisation de [reflexivity], qui elle-même effectue des
    simplifications -- i.e., la première simplifie [beq_nat (S n' + 1) 0] to
    [false] en réécrivant d'abord [(S n' + 1)] en [S (n' + 1)], ensuite en
    dépliant [beq_nat] et enfin en simplifiant le filtrage.

    Marquer les cas avec des tirets est complètement optionnel. Si les tirets ne
    sont pas présents, Coq demande seulement de prouver chaque sous-but de façon
    séquentielle, un à la fois. Cependant, c'est une bonne pratique que
    d'utiliser des tirets. Pour la seule raison qu'ils rendent apparente la
    structure de la preuve et la rendent plus lisible. De plus, les tirets
    indiquent à Coq de s'assurer qu'un but est prouvé avant de vérifier le
    suivant, évitant ainsi que les preuves des différents sous-buts se
    mélangent. Ce point est particulièrement important pour les grands
    développements où les preuves mal structurées conduisent à de longues
    sessions de corrections.

    Il n'y a pas de règles strictes qui indiquent comment les preuves doivent
    être formatées en Coq -- en particulier, l'endroit où les lignes doivent
    être scindées et comment les sections d'une preuve doivent être indentées
    pour indiquer leur structure emboitée. Cependant, si les endroits où de
    multiples sous-buts sont générés sont marqués avec des tirets en début de
    ligne, alors la preuve sera plus lisible qu'importe les choix de
    présentation qui seront faits par ailleurs.

    C'est aussi le bon moment pour donner un autre conseil évident à propos de
    la longueur des lignes. Les débutants en Coq ont parfois des tendances
    extrêmes, soit ils écrivent chaque tactique sur une ligne ou bien ils
    écrivent toute la preuve sur une ligne. Un bon style de présentation est,
    bien sûr, entre les deux. Une convention raisonnable est de ne pas dépasser
    80 caractères par ligne.

    La tactique [destruct] peut être utilisée avec n'importe quel structure de
    donnée définie inductivement. Par exemple, on l'utilise ci-après pour
    prouver que la négation booléenne est involutive -- i.e., la négation est
    son propre inverse. *)

Theorem negb_involutive : forall b : bool,
  negb (negb b) = b.
Proof.
  intros b. destruct b.
  - reflexivity.
  - reflexivity.  Qed.

(** Remarquez qu'ici [destruct] n'a pas de clause [as] car aucun des
    sous-cas de [destruct] n'utilise de variables, il n'y a donc pas
    besoin d'indiquer de noms (on aurait pu écrire [as [|]] ou [as []]
    ce qui est inutile). En fait, on peut omettre la clause [as] pour
    toutes les utilisations de [destruct], dans ce cas Coq remplit les
    noms de variables automatiquement. C'est, en général, considéré
    comme un mauvais style car Coq fait souvent des choix de noms de
    variables difficiles à utiliser.

    Il est parfois utile d'appeler [destruct] à l'intérieur d'un
    sous-but, générant ainsi des obligations de preuve
    supplémentaires. Dans ce cas, on utilise des tirets différents
    pour marquer les buts des différents "niveaux". Par exemple : *)

Theorem andb_commutative : forall b c, andb b c = andb c b.
Proof.
  intros b c. destruct b.
  - destruct c.
    + reflexivity.
    + reflexivity.
  - destruct c.
    + reflexivity.
    + reflexivity.
Qed.

(** Chaque paire d'appels à [reflexivity] correspond au sous-but qui a
    été généré après l'exécution de [destruct c] de la ligne juste
    au-dessus. *)

(** En plus de [-] et [+], on peut utiliser [*] (l'astérisque) comme
    troisième sorte de tirets. On peut aussi entourer les sous-preuves
    avec des accolades, cela est utile dans le cas où la preuve
    considérée génère plus de trois niveaux de sous-buts : *)

Theorem andb_commutative' : forall b c, andb b c = andb c b.
Proof.
  intros b c. destruct b.
  { destruct c.
    { reflexivity. }
    { reflexivity. } }
  { destruct c.
    { reflexivity. }
    { reflexivity. } }
Qed.

(** Comme les accolades marquent à la fois le début et la fin d'une
    preuve, elles peuvent être utilisées pour des niveaux multiples de
    sous-buts, comme l'exemple le montre. De plus, les accolades
    permettent aussi de réutiliser les mêmes formes de tirets à
    différents niveaux dans une preuve : *)

Theorem andb3_exchange :
  forall b c d, andb (andb b c) d = andb (andb b d) c.
Proof.
  intros b c d. destruct b.
  - destruct c.
    { destruct d.
      - reflexivity.
      - reflexivity. }
    { destruct d.
      - reflexivity.
      - reflexivity. }
  - destruct c.
    { destruct d.
      - reflexivity.
      - reflexivity. }
    { destruct d.
      - reflexivity.
      - reflexivity. }
Qed.

(** Avant de terminer ce chapitre, voici une autre facilité de
    notation. Comme vous avez pu le remarquer, plusieurs preuves
    effectuent une analyse de cas sur une variable juste après son
    introduction :

    intros x y. destruct y as [|y].
    
    Ce motif est tellement courant que Coq fournit un raccourci pour
    cela. On peut effectuer une analyse de cas sur une variable en
    l'introduisant avec un motif d'introduction à la place d'un nom de
    variable. Par exemple, voici une preuve plus courte du théorème
    précédent [plus_1_neq_0]. *) 

Theorem plus_1_neq_0' : forall n : nat,
  beq_nat (n + 1) 0 = false.
Proof.
  intros [|n].
  - reflexivity.
  - reflexivity.  Qed.

(** S'il n'y pas d'arguments à nommer, on peut écrire seulement
    [[]]. *)

Theorem andb_commutative'' :
  forall b c, andb b c = andb c b.
Proof.
  intros [] [].
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
Qed.

(** **** Exercice: ** (andb_true_elim2)  *)
(** Prouvez l'assertion suivante en marquant les cas (et les sous-cas)
    avec des tirets quand vous utilisez [destruct]. *)

Theorem andb_true_elim2 : forall b c : bool,
  andb b c = true -> c = true.
Proof.
  intros b c H.
  destruct b.
    - destruct c.
      + reflexivity.  
      + simpl in H. 
        *exact H.
    - destruct c.
      + reflexivity.
      + simpl in H. 
        *exact H.      
Qed.
(** [] *)

(** **** Exercice: * (zero_nbeq_plus_1)  *)
Theorem zero_nbeq_plus_1 : forall n : nat,
  beq_nat 0 (n + 1) = false.
Proof.
  intros [|n'].
    - simpl. 
      + reflexivity.
    - simpl.
      + reflexivity.
Qed.
  (* Remplir ici *) 
(** [] *)

(* ################################################################# *)
(** * Exercices supplémentaires *)

(** **** Exercice: ** (boolean_functions)  *)
(** Utilisez les tactiques que vous avez apprises pour prouver les
    théorèmes suivants à propos des fonctions booléennes.*)

Theorem identity_fn_applied_twice :
  forall (f : bool -> bool),
  (forall (x : bool), f x = x) ->
  forall (b : bool), f (f b) = b.
Proof.
  intros.
  destruct b.
    - rewrite -> H.
      + rewrite -> H. 
      + reflexivity.
  (* Remplir ici *) Admitted.

(** Maintenant écrivez et prouver un théorème
    [negation_fn_applied_twice] similaire au théorème précédent, mais
    où la seconde hypothèse dit que la fonction [f] a la propriété [f
    x = negb x]. *)

(* Remplir ici *)
(** [] *)

(** **** Exercice: ***, optionnel (andb_eq_orb)  *)
(** Prouvez le théorème suivant. Indication : La preuve peut devenir
    compliquée selon l'approche qui vous suivie. Vous allez devoir
    probablement utiliser [destruct] et [rewrite], mais tout
    déstructurer n'est pas la meilleure idée. *)

Theorem andb_eq_orb :
  forall (b c : bool),
  (andb b c = orb b c) ->
  b = c.
Proof.
  (* Remplir ici *) Admitted.

(** [] *)



