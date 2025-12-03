; Full Propositional Resolution Prover for MeTTa-Compiler
; Uses Cons/Nil for dynamic clause lists

; =============================================================================
; LIST UTILITIES
; =============================================================================

(= (length Nil) 0)
(= (length (Cons $h $t)) (+ 1 (length $t)))

(= (append Nil $list) $list)
(= (append (Cons $h $t) $list) (Cons $h (append $t $list)))

(= (member $x (Cons $x $t)) true)
(= (member $x (Cons $h $t)) (member $x $t))
(= (member $x Nil) false)

; =============================================================================
; LITERAL OPERATIONS
; =============================================================================

; Check if two literals are complementary
(= (complementary (neg $x) $x) true)
(= (complementary $x (neg $x)) true)
(= (complementary $x $y) false)

; Test
!(complementary p (neg p))
!(complementary p q)

; =============================================================================
; CLAUSE REPRESENTATION
; =============================================================================
; Clause = Cons-list of literals
; Example: (Cons p (Cons q Nil)) represents {p ∨ q}

; Check if clause is empty (contradiction)
(= (is-empty-clause Nil) true)
(= (is-empty-clause (Cons $h $t)) false)

; Unit clause = single literal
(= (is-unit (Cons $lit Nil)) (unit $lit))
(= (is-unit $clause) not-unit)

; Test
!(is-empty-clause Nil)
!(is-empty-clause (Cons p Nil))
!(is-unit (Cons p Nil))
!(is-unit (Cons p (Cons q Nil)))

; =============================================================================
; RESOLUTION
; =============================================================================

; Find complementary literal in clause
(= (find-comp $lit (Cons $h $t))
   (if (== (complementary $lit $h) true)
       (found $h $t)
       (case (find-comp $lit $t)
         ((not-found not-found)
          ((found $comp $rest) (found $comp (Cons $h $rest)))))))

(= (find-comp $lit Nil) not-found)

; Resolve two clauses
(= (resolve-clauses (Cons $lit $rest1) $clause2)
   (case (find-comp $lit $clause2)
     ((not-found
       (case $rest1
         ((Nil no-resolution)
          ($_ (resolve-clauses $rest1 $clause2)))))
      ((found $comp $rest2)
       (resolvent (append $rest1 $rest2))))))

(= (resolve-clauses Nil $clause2) no-resolution)

; Test resolution
!(resolve-clauses (Cons p (Cons q Nil)) (Cons (neg p) (Cons r Nil)))
!(resolve-clauses (Cons p Nil) (Cons (neg p) Nil))

; =============================================================================
; SIMPLE EXAMPLES
; =============================================================================

; Example 1: {p, ¬p} → empty clause (UNSAT)
!(let $c1 (Cons p Nil)
    (let $c2 (Cons (neg p) Nil)
      (resolve-clauses $c1 $c2)))

; Example 2: {p∨q, ¬p∨r} → {q∨r}
!(let $c1 (Cons p (Cons q Nil))
    (let $c2 (Cons (neg p) (Cons r Nil))
      (resolve-clauses $c1 $c2)))

; Example 3: No resolution possible
!(let $c1 (Cons p (Cons q Nil))
    (let $c2 (Cons r (Cons s Nil))
      (resolve-clauses $c1 $c2)))
