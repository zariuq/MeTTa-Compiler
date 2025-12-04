;; =============================================================================
;; FOL Term Representation (Ported from PyRes terms.py)
;; =============================================================================
;;
;; Term datatype (following PyRes but with explicit constructors):
;; - Variables: (Var "X"), (Var "Y"), (Var "_")
;; - Constants: (Const "a"), (Const "b")
;; - Functions: (Fun "f" (arg1 arg2 ...))
;;
;; PyRes uses nested lists: "X", ["a"], ["f", "X", ["a"]]
;; We use explicit tagged unions for type safety and clarity.
;;
;; Design: Single-answer deterministic functions (no non-determinism)
;; =============================================================================

;; -----------------------------------------------------------------------------
;; Term Type Predicates
;; -----------------------------------------------------------------------------

;; Check if term is a variable
;; termIsVar : Term -> Bool
(= (term-is-var (Var $name)) True)
(= (term-is-var (Const $name)) False)
(= (term-is-var (Fun $f $args)) False)

;; Check if term is a constant
(= (term-is-const (Const $name)) True)
(= (term-is-const (Var $name)) False)
(= (term-is-const (Fun $f $args)) False)

;; Check if term is a compound (function application)
(= (term-is-compound (Fun $f $args)) True)
(= (term-is-compound (Var $name)) False)
(= (term-is-compound (Const $name)) False)

;; -----------------------------------------------------------------------------
;; Term Accessors
;; -----------------------------------------------------------------------------

;; Get function symbol from compound term
;; Precondition: term-is-compound t = True
;; termFunc : Term -> String
(= (term-func (Fun $f $_)) $f)

;; Get arguments from compound term
;; Precondition: term-is-compound t = True
;; termArgs : Term -> List<Term>
(= (term-args (Fun $_ $args)) $args)

;; Get variable name
;; Precondition: term-is-var t = True
(= (term-var-name (Var $name)) $name)

;; Get constant name
;; Precondition: term-is-const t = True
(= (term-const-name (Const $name)) $name)

;; -----------------------------------------------------------------------------
;; Term Equality
;; -----------------------------------------------------------------------------

;; Structural equality for terms
;; termEqual : Term -> Term -> Bool
(= (term-equal (Var $x) (Var $y))
   (if (== $x $y) True False))

(= (term-equal (Const $a) (Const $b))
   (if (== $a $b) True False))

(= (term-equal (Fun $f1 $args1) (Fun $f2 $args2))
   (if (== $f1 $f2)
       (term-list-equal $args1 $args2)
       False))

;; Different term types are never equal
(= (term-equal (Var $v) (Const $c)) False)
(= (term-equal (Var $v) (Fun $f $args)) False)
(= (term-equal (Const $c) (Var $v)) False)
(= (term-equal (Const $c) (Fun $f $args)) False)
(= (term-equal (Fun $f $args) (Var $v)) False)
(= (term-equal (Fun $f $args) (Const $c)) False)

;; List equality for term arguments
;; termListEqual : List<Term> -> List<Term> -> Bool
(= (term-list-equal Nil Nil) True)
(= (term-list-equal Nil (Cons $h $t)) False)
(= (term-list-equal (Cons $h $t) Nil) False)
(= (term-list-equal (Cons $h1 $t1) (Cons $h2 $t2))
   (if (== (term-equal $h1 $h2) True)
       (term-list-equal $t1 $t2)
       False))

;; -----------------------------------------------------------------------------
;; Term Properties
;; -----------------------------------------------------------------------------

;; Check if term is ground (contains no variables)
;; termIsGround : Term -> Bool
(= (term-is-ground (Var $_)) False)
(= (term-is-ground (Const $_)) True)
(= (term-is-ground (Fun $_ $args))
   (term-list-is-ground $args))

;; Helper: check if list of terms is ground
(= (term-list-is-ground Nil) True)
(= (term-list-is-ground (Cons $h $t))
   (if (== (term-is-ground $h) True)
       (term-list-is-ground $t)
       False))

;; Collect all variables in a term (returns list, may have duplicates)
;; termCollectVars : Term -> List<String>
(= (term-collect-vars (Var $x))
   (Cons $x Nil))

(= (term-collect-vars (Const $_))
   Nil)

(= (term-collect-vars (Fun $_ $args))
   (term-list-collect-vars $args))

;; Helper: collect variables from term list
(= (term-list-collect-vars Nil) Nil)
(= (term-list-collect-vars (Cons $h $t))
   (append-lists (term-collect-vars $h)
                 (term-list-collect-vars $t)))

;; Collect all function symbols (returns list with duplicates)
;; termCollectFuns : Term -> List<String>
(= (term-collect-funs (Var $_)) Nil)
(= (term-collect-funs (Const $c)) (Cons $c Nil))
(= (term-collect-funs (Fun $f $args))
   (Cons $f (term-list-collect-funs $args)))

;; Helper: collect functions from term list
(= (term-list-collect-funs Nil) Nil)
(= (term-list-collect-funs (Cons $h $t))
   (append-lists (term-collect-funs $h)
                 (term-list-collect-funs $t)))

;; -----------------------------------------------------------------------------
;; Term Weight (for heuristics)
;; -----------------------------------------------------------------------------

;; Compute term weight (sum of function/variable weights)
;; termWeight : Term -> Int -> Int -> Int
;; Parameters: term, function-weight, variable-weight
(= (term-weight (Var $name) $fweight $vweight) $vweight)
(= (term-weight (Const $name) $fweight $vweight) $fweight)
(= (term-weight (Fun $f $args) $fweight $vweight)
   (+ $fweight (term-list-weight $args $fweight $vweight)))

;; Helper: sum weights of term list
(= (term-list-weight Nil $fweight $vweight) 0)
(= (term-list-weight (Cons $h $t) $fweight $vweight)
   (+ (term-weight $h $fweight $vweight)
      (term-list-weight $t $fweight $vweight)))

;; Standard weight: functions=1, variables=1
(= (term-std-weight $term)
   (term-weight $term 1 1))

;; -----------------------------------------------------------------------------
;; Utility Functions
;; -----------------------------------------------------------------------------

;; Append two lists (deterministic, tail-recursive friendly)
(= (append-lists Nil $l2) $l2)
(= (append-lists (Cons $h $t) $l2)
   (Cons $h (append-lists $t $l2)))

;; Length of list
(= (list-length Nil) 0)
(= (list-length (Cons $_ $t))
   (+ 1 (list-length $t)))

;; =============================================================================
;; Examples (for documentation)
;; =============================================================================

;; Variables: (Var "X"), (Var "Y"), (Var "Z")
;; Constants: (Const "a"), (Const "b"), (Const "c")
;; Functions:
;;   (Fun "f" (Cons (Var "X") Nil))                     ; f(X)
;;   (Fun "g" (Cons (Const "a") (Cons (Const "b") Nil))) ; g(a,b)
;;   (Fun "h" (Cons (Var "X")
;;                  (Cons (Fun "f" (Cons (Var "Y") Nil))
;;                        Nil)))                        ; h(X, f(Y))
