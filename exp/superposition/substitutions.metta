;; =============================================================================
;; Substitutions for First-Order Terms
;; =============================================================================
;;
;; A substitution σ is a mapping from variables to terms.
;; We represent it as a list of bindings: (Cons (Binding var term) rest)
;;
;; Design: Deterministic, functional (immutable)
;; Based on: PyRes substitutions.py
;; =============================================================================

;; -----------------------------------------------------------------------------
;; Substitution Data Structure
;; -----------------------------------------------------------------------------

;; Binding: (Binding var-name term)
;; Example: (Binding "X" (Const "a"))  means X <- a

;; Substitution: List of bindings
;; Empty substitution: Nil
;; Example: (Cons (Binding "X" (Const "a"))
;;                (Cons (Binding "Y" (Var "Z")) Nil))

;; -----------------------------------------------------------------------------
;; Lookup and Binding
;; -----------------------------------------------------------------------------

;; Look up a variable in a substitution
;; substLookup : String -> Subst -> Term
;; Returns the binding if found, or (Var var-name) if not bound
(= (subst-lookup $var Nil)
   (Var $var))
(= (subst-lookup $var (Cons (Binding $v $term) $rest))
   (if (== $var $v)
       $term
       (subst-lookup $var $rest)))

;; Check if a variable is bound in substitution
;; substIsBound : String -> Subst -> Bool
(= (subst-is-bound $var Nil) False)
(= (subst-is-bound $var (Cons (Binding $v $term) $rest))
   (if (== $var $v)
       True
       (subst-is-bound $var $rest)))

;; -----------------------------------------------------------------------------
;; Apply Substitution
;; -----------------------------------------------------------------------------

;; Apply substitution to a term
;; substApply : Subst -> Term -> Term
(= (subst-apply $subst (Var $name))
   (subst-lookup $name $subst))

(= (subst-apply $subst (Const $name))
   (Const $name))

(= (subst-apply $subst (Fun $f $args))
   (Fun $f (subst-apply-list $subst $args)))

;; Helper: apply substitution to a list of terms
(= (subst-apply-list $subst Nil) Nil)
(= (subst-apply-list $subst (Cons $h $t))
   (Cons (subst-apply $subst $h)
         (subst-apply-list $subst $t)))

;; -----------------------------------------------------------------------------
;; Compose Substitutions
;; -----------------------------------------------------------------------------

;; Compose a new binding into an existing substitution
;; substComposeBinding : Subst -> (Binding var term) -> Subst
;;
;; This applies the new binding to all existing bindings,
;; then adds the new binding if the variable isn't already bound
(= (subst-compose-binding $subst (Binding $var $term))
   (subst-compose-binding-helper
      (subst-apply-to-bindings $subst (Binding $var $term))
      $var
      $term))

;; Apply a new binding to all existing bindings in a substitution
(= (subst-apply-to-bindings Nil $new-binding) Nil)
(= (subst-apply-to-bindings (Cons (Binding $v $t) $rest) $new-binding)
   (Cons (Binding $v (subst-apply (Cons $new-binding Nil) $t))
         (subst-apply-to-bindings $rest $new-binding)))

;; Add the new binding if variable not already bound
(= (subst-compose-binding-helper $updated-subst $var $term)
   (if (subst-is-bound $var $updated-subst)
       $updated-subst
       (Cons (Binding $var $term) $updated-subst)))

;; -----------------------------------------------------------------------------
;; Utility Functions
;; -----------------------------------------------------------------------------

;; Create empty substitution
(= (subst-empty) Nil)

;; Create substitution from single binding
(= (subst-single $var $term)
   (Cons (Binding $var $term) Nil))

;; Count bindings in substitution (for debugging)
(= (subst-size Nil) 0)
(= (subst-size (Cons $binding $rest))
   (+ 1 (subst-size $rest)))
