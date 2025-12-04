; Unification: Robinson's MGU with occurs check (from PyRes)

(= (occurs-check $x (Var $name))
   (== $x $name))

(= (occurs-check $x (Const $name))
   False)

(= (occurs-check $x (Fun $f $args))
   (occurs-check-list $x $args))

(= (occurs-check-list $x Nil) False)
(= (occurs-check-list $x (Cons $h $t))
   (if (occurs-check $x $h)
       True
       (occurs-check-list $x $t)))

(= (mgu $t1 $t2)
   (mgu-term-list (Cons $t1 Nil)
                  (Cons $t2 Nil)
                  (subst-empty)))

(= (mgu-term-list Nil Nil $subst)
   (Success $subst))

(= (mgu-term-list (Cons $t1 $rest1) (Cons $t2 $rest2) $subst)
   (mgu-step $t1 $t2 $rest1 $rest2 $subst))

; Single dispatch function using nested type checks
(= (mgu-step $t1 $t2 $rest1 $rest2 $subst)
   (if (term-is-var $t1)
       (if (term-is-var $t2)
           (if (== (term-var-name $t1) (term-var-name $t2))
               (mgu-term-list $rest1 $rest2 $subst)
               (mgu-bind (term-var-name $t1) $t2 $rest1 $rest2 $subst))
           (mgu-bind (term-var-name $t1) $t2 $rest1 $rest2 $subst))
       (if (term-is-var $t2)
           (mgu-bind (term-var-name $t2) $t1 $rest1 $rest2 $subst)
           (if (term-is-const $t1)
               (if (term-is-const $t2)
                   (if (== (term-const-name $t1) (term-const-name $t2))
                       (mgu-term-list $rest1 $rest2 $subst)
                       (Failure))
                   (Failure))
               (if (term-is-compound $t1)
                   (if (term-is-compound $t2)
                       (if (== (term-func $t1) (term-func $t2))
                           (mgu-term-list (append-lists (term-args $t1) $rest1)
                                          (append-lists (term-args $t2) $rest2)
                                          $subst)
                           (Failure))
                       (Failure))
                   (Failure))))))

(= (mgu-bind $x $t $rest1 $rest2 $subst)
   (if (occurs-check $x $t)
       (Failure)
       (mgu-bind-apply $x $t $rest1 $rest2 $subst)))

(= (mgu-bind-apply $x $t $rest1 $rest2 $subst)
   (let $new-binding (Binding $x $t)
     (let $new-subst (subst-single $x $t)
       (mgu-term-list
          (subst-apply-list $new-subst $rest1)
          (subst-apply-list $new-subst $rest2)
          (subst-compose-binding $subst $new-binding)))))

(= (unify-success (Success $subst)) True)
(= (unify-success (Failure)) False)

(= (unify-get-subst (Success $subst)) $subst)
(= (unify-get-subst (Failure)) Nil)
