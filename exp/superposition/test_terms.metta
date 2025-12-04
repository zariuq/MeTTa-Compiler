;; =============================================================================
;; Unit Tests for Term Operations
;; =============================================================================

!(import! &self terms.metta rel)

; Test helper - returns () on pass, prints and returns error details on fail
(= (test $name $actual $expected)
   (if (== $actual $expected)
       (())
       (println! (format "FAIL: {}: expected {}, got {}" ($name $expected $actual)))))

;; =============================================================================
;; Test 1: Type Predicates
;; =============================================================================

!(test "term-is-var on Var"
   (term-is-var (Var "X")) True)

!(test "term-is-var on Const"
   (term-is-var (Const "a")) False)

!(test "term-is-var on Fun"
   (term-is-var (Fun "f" Nil)) False)

!(test "term-is-const on Const"
   (term-is-const (Const "a")) True)

!(test "term-is-const on Var"
   (term-is-const (Var "X")) False)

!(test "term-is-compound on Fun"
   (term-is-compound (Fun "f" (Cons (Var "X") Nil))) True)

!(test "term-is-compound on Var"
   (term-is-compound (Var "X")) False)

;; =============================================================================
;; Test 2: Accessors
;; =============================================================================

!(test "term-func"
   (term-func (Fun "f" Nil)) "f")

!(test "term-args"
   (term-args (Fun "f" (Cons (Var "X") Nil)))
   (Cons (Var "X") Nil))

!(test "term-var-name"
   (term-var-name (Var "X")) "X")

!(test "term-const-name"
   (term-const-name (Const "a")) "a")

;; =============================================================================
;; Test 3: Term Equality
;; =============================================================================

!(test "term-equal: same vars"
   (term-equal (Var "X") (Var "X")) True)

!(test "term-equal: different vars"
   (term-equal (Var "X") (Var "Y")) False)

!(test "term-equal: same consts"
   (term-equal (Const "a") (Const "a")) True)

!(test "term-equal: different consts"
   (term-equal (Const "a") (Const "b")) False)

!(test "term-equal: var vs const"
   (term-equal (Var "X") (Const "a")) False)

!(test "term-equal: same functions"
   (term-equal
     (Fun "f" (Cons (Var "X") Nil))
     (Fun "f" (Cons (Var "X") Nil)))
   True)

!(test "term-equal: different function symbols"
   (term-equal
     (Fun "f" (Cons (Var "X") Nil))
     (Fun "g" (Cons (Var "X") Nil)))
   False)

!(test "term-equal: different args"
   (term-equal
     (Fun "f" (Cons (Var "X") Nil))
     (Fun "f" (Cons (Var "Y") Nil)))
   False)

;; =============================================================================
;; Test 4: Ground Terms
;; =============================================================================

!(test "term-is-ground: variable"
   (term-is-ground (Var "X")) False)

!(test "term-is-ground: constant"
   (term-is-ground (Const "a")) True)

!(test "term-is-ground: f(a)"
   (term-is-ground (Fun "f" (Cons (Const "a") Nil))) True)

!(test "term-is-ground: f(X)"
   (term-is-ground (Fun "f" (Cons (Var "X") Nil))) False)

!(test "term-is-ground: f(a,b)"
   (term-is-ground
     (Fun "f" (Cons (Const "a") (Cons (Const "b") Nil))))
   True)

!(test "term-is-ground: f(a,X)"
   (term-is-ground
     (Fun "f" (Cons (Const "a") (Cons (Var "X") Nil))))
   False)

;; =============================================================================
;; Test 5: Collect Variables
;; =============================================================================

!(test "collect-vars: variable"
   (term-collect-vars (Var "X"))
   (Cons "X" Nil))

!(test "collect-vars: constant"
   (term-collect-vars (Const "a"))
   Nil)

!(test "collect-vars: f(X)"
   (term-collect-vars (Fun "f" (Cons (Var "X") Nil)))
   (Cons "X" Nil))

!(test "collect-vars: f(X,Y)"
   (term-collect-vars
     (Fun "f" (Cons (Var "X") (Cons (Var "Y") Nil))))
   (Cons "X" (Cons "Y" Nil)))

;; Note: duplicates allowed
!(test "collect-vars: f(X,X)"
   (term-collect-vars
     (Fun "f" (Cons (Var "X") (Cons (Var "X") Nil))))
   (Cons "X" (Cons "X" Nil)))

;; =============================================================================
;; Test 6: Collect Functions
;; =============================================================================

!(test "collect-funs: variable"
   (term-collect-funs (Var "X"))
   Nil)

!(test "collect-funs: constant"
   (term-collect-funs (Const "a"))
   (Cons "a" Nil))

!(test "collect-funs: f(a)"
   (term-collect-funs (Fun "f" (Cons (Const "a") Nil)))
   (Cons "f" (Cons "a" Nil)))

!(test "collect-funs: g(f(a),b)"
   (term-collect-funs
     (Fun "g" (Cons (Fun "f" (Cons (Const "a") Nil))
                    (Cons (Const "b") Nil))))
   (Cons "g" (Cons "f" (Cons "a" (Cons "b" Nil)))))

;; =============================================================================
;; Test 7: Term Weight
;; =============================================================================

!(test "term-weight: variable"
   (term-weight (Var "X") 1 1) 1)

!(test "term-weight: constant"
   (term-weight (Const "a") 1 1) 1)

!(test "term-weight: f(X)"
   (term-weight (Fun "f" (Cons (Var "X") Nil)) 1 1)
   2)  ; f=1 + X=1

!(test "term-weight: f(a,b)"
   (term-weight
     (Fun "f" (Cons (Const "a") (Cons (Const "b") Nil)))
     1 1)
   3)  ; f=1 + a=1 + b=1

!(test "term-weight: g(f(X),a)"
   (term-weight
     (Fun "g" (Cons (Fun "f" (Cons (Var "X") Nil))
                    (Cons (Const "a") Nil)))
     1 1)
   4)  ; g=1 + (f=1 + X=1) + a=1

!(test "term-std-weight: f(X,Y)"
   (term-std-weight
     (Fun "f" (Cons (Var "X") (Cons (Var "Y") Nil))))
   3)  ; f=1 + X=1 + Y=1

!(println! "\nAll 38 tests completed!")
