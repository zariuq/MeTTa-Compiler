; Integration tests for car-atom, cdr-atom, cons-atom, decons-atom
; Run with: mettatron examples/test_car_cdr_atom.metta

; === Basic car-atom tests ===
!(car-atom (a b c))
; Expected: [a]

!(car-atom (hello))
; Expected: [hello]

; === Basic cdr-atom tests ===
!(cdr-atom (a b c))
; Expected: [(b c)]

!(cdr-atom (a))
; Expected: [()]

; === Basic cons-atom tests ===
!(cons-atom x (y z))
; Expected: [(x y z)]

!(cons-atom first ())
; Expected: [(first)]

!(cons-atom head ())
; Expected: [(head)]

; === Basic decons-atom tests ===
!(decons-atom (a b c))
; Expected: [(a (b c))]

!(decons-atom (single))
; Expected: [(single ())]

; === Composition tests ===

; Test: (cons-atom (car-atom X) (cdr-atom X)) = X
!(let $head (car-atom (p q r))
   (cons-atom $head (cdr-atom (p q r))))
; Expected: [(p q r)]

; Test: using car-atom/cdr-atom to extract and recombine
!(let $first (car-atom (alpha beta gamma))
   (let $rest (cdr-atom (alpha beta gamma))
      (cons-atom $first $rest)))
; Expected: [(alpha beta gamma)]

; === Resolution prover patterns ===

; Negation pattern: get literal, negate it, cons back
!(let $lit (car-atom (p q r))
   (let $rest (cdr-atom (p q r))
      (cons-atom (neg $lit) $rest)))
; Expected: [((neg p) q r)]

; Append-like operation using car/cdr
!(let $h (car-atom (a b))
   (cons-atom $h (c d)))
; Expected: [(a c d)]

; === Working with Cons/Nil constructors ===

; IMPORTANT: car-atom/cdr-atom/cons-atom are SPECIAL FORMS with unevaluated arguments
; To compose them, use 'let' to bind intermediate results (same as Hyperon Experimental MeTTa)

; ❌ This DOES NOT WORK - nested call is not evaluated:
; !(cons-atom p (cdr-atom (b c)))  → [(p cdr-atom (b c))]

; ✅ This WORKS - use let to evaluate and bind:
!(let $rest (cdr-atom (a b c))
   (cons-atom p $rest))
; Expected: [(p b c)]

; Convert between representations
!(let $sexpr (cons-atom a (cons-atom b ()))
   (car-atom $sexpr))
; Expected: [a]

!(let $sexpr (cons-atom a (cons-atom b (cons-atom c ())))
   (cdr-atom $sexpr))
; Expected: [(b c)]

; === Edge cases ===

; Empty list error
; !(car-atom ())
; Should error: "car-atom: expression is empty"

; !(cdr-atom ())
; Should error: "cdr-atom: expression is empty"

; Non-list error
; !(car-atom 42)
; Should error: "car-atom: expected expression"

; !(cons-atom a 42)
; Should error: "cons-atom: tail must be expression or Nil"

; === Nested operations ===

; Get second element using car-cdr-car
!(car-atom (cdr-atom (first second third)))
; Expected: [second]

; Get third element
!(car-atom (cdr-atom (cdr-atom (one two three four))))
; Expected: [three]

; Build nested structure
!(cons-atom (cons-atom a (b c)) (d e))
; Expected: [((a b c) d e)]

; === Resolution-style clause manipulation ===

; Simulate removing first literal from clause
!(let $clause (p q r s)
   (cdr-atom $clause))
; Expected: [(q r s)]

; Simulate adding literal to front of clause
!(let $lit (neg p)
   (let $clause (q r)
      (cons-atom $lit $clause)))
; Expected: [((neg p) q r)]

; Simulate deconstructing for resolution
!(let $clause1 (p q r)
   (let $parts (decons-atom $clause1)
      $parts))
; Expected: [(p (q r))]
