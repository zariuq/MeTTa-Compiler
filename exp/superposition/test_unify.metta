; Unit Tests for Substitutions and Unification

!(import! &self terms.metta rel)
!(import! &self substitutions.metta rel)
!(import! &self unification.metta rel)

; Test helper - returns () on pass, prints and returns error details on fail
(= (test $name $actual $expected)
   (if (== $actual $expected)
       (())
       (println! (format "FAIL: {}: expected {}, got {}" ($name $expected $actual)))))

;
;; Test 1: Substitution Lookup
;

!(test "lookup: unbound variable"
   (subst-lookup "X" Nil)
   (Var "X"))

!(test "lookup: bound variable"
   (subst-lookup "X" (Cons (Binding "X" (Const "a")) Nil))
   (Const "a"))

!(test "lookup: variable not in substitution"
   (subst-lookup "Y" (Cons (Binding "X" (Const "a")) Nil))
   (Var "Y"))

;
;; Test 2: Substitution Application
;

!(test "apply: variable bound to constant"
   (subst-apply (Cons (Binding "X" (Const "a")) Nil) (Var "X"))
   (Const "a"))

!(test "apply: constant unchanged"
   (subst-apply (Cons (Binding "X" (Const "a")) Nil) (Const "b"))
   (Const "b"))

!(test "apply: function with var"
   (subst-apply
      (Cons (Binding "X" (Const "a")) Nil)
      (Fun "f" (Cons (Var "X") Nil)))
   (Fun "f" (Cons (Const "a") Nil)))

!(test "apply: nested function"
   (subst-apply
      (Cons (Binding "X" (Const "a"))
            (Cons (Binding "Y" (Const "b")) Nil))
      (Fun "f" (Cons (Var "X") (Cons (Var "Y") Nil))))
   (Fun "f" (Cons (Const "a") (Cons (Const "b") Nil))))

;
;; Test 3: Occurs Check
;

!(test "occurs-check: X in X"
   (occurs-check "X" (Var "X"))
   True)

!(test "occurs-check: X not in Y"
   (occurs-check "X" (Var "Y"))
   False)

!(test "occurs-check: X not in constant"
   (occurs-check "X" (Const "a"))
   False)

!(test "occurs-check: X in f(X)"
   (occurs-check "X" (Fun "f" (Cons (Var "X") Nil)))
   True)

!(test "occurs-check: X not in f(Y)"
   (occurs-check "X" (Fun "f" (Cons (Var "Y") Nil)))
   False)

!(test "occurs-check: X in f(a,X)"
   (occurs-check "X" (Fun "f" (Cons (Const "a") (Cons (Var "X") Nil))))
   True)

;
;; Test 4: Basic Unification
;

!(test "mgu: X=a"
   (unify-success (mgu (Var "X") (Const "a")))
   True)

!(test "mgu: X=X (solved)"
   (unify-success (mgu (Var "X") (Var "X")))
   True)

!(test "mgu: X=Y"
   (unify-success (mgu (Var "X") (Var "Y")))
   True)

!(test "mgu: a=a"
   (unify-success (mgu (Const "a") (Const "a")))
   True)

!(test "mgu: a=b (fail)"
   (unify-success (mgu (Const "a") (Const "b")))
   False)

;
;; Test 5: Function Unification
;

!(test "mgu: f(a)=f(a)"
   (unify-success (mgu
      (Fun "f" (Cons (Const "a") Nil))
      (Fun "f" (Cons (Const "a") Nil))))
   True)

!(test "mgu: f(X)=f(a)"
   (unify-success (mgu
      (Fun "f" (Cons (Var "X") Nil))
      (Fun "f" (Cons (Const "a") Nil))))
   True)

!(test "mgu: f(X,a)=f(b,Y)"
   (unify-success (mgu
      (Fun "f" (Cons (Var "X") (Cons (Const "a") Nil)))
      (Fun "f" (Cons (Const "b") (Cons (Var "Y") Nil)))))
   True)

!(test "mgu: f(a)=g(a) (structural fail)"
   (unify-success (mgu
      (Fun "f" (Cons (Const "a") Nil))
      (Fun "g" (Cons (Const "a") Nil))))
   False)

;
;; Test 6: Occurs Check in Unification
;

!(test "mgu: X=f(X) (occurs check fail)"
   (unify-success (mgu
      (Var "X")
      (Fun "f" (Cons (Var "X") Nil))))
   False)

!(test "mgu: X=f(g(X)) (occurs check fail)"
   (unify-success (mgu
      (Var "X")
      (Fun "f" (Cons (Fun "g" (Cons (Var "X") Nil)) Nil))))
   False)

;
;; Test 7: Complex Unification
;

!(test "mgu: f(X,g(a))=f(X,Y)"
   (unify-success (mgu
      (Fun "f" (Cons (Var "X") (Cons (Fun "g" (Cons (Const "a") Nil)) Nil)))
      (Fun "f" (Cons (Var "X") (Cons (Var "Y") Nil)))))
   True)

!(test "mgu: f(X,g(a))=f(X,X) (succeeds, X binds to g(a))"
   (unify-success (mgu
      (Fun "f" (Cons (Var "X") (Cons (Fun "g" (Cons (Const "a") Nil)) Nil)))
      (Fun "f" (Cons (Var "X") (Cons (Var "X") Nil)))))
   True)

!(println! "\nAll unification tests completed!")
