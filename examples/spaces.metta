; ============================================================================
; Space Management Examples - Compact & Elegant
; ============================================================================

; ----------------------------------------------------------------------------
; Basic Space Operations
; ----------------------------------------------------------------------------
; Create and bind a knowledge base
!(bind-space &kb (new-space))
!(add-atom &kb (person alice 30))
!(add-atom &kb (person bob 25))
!(match &kb (person $name $age) $name)
; Returns: [alice, bob]

; &self is always available (the default space)
!(match &self (= $lhs $rhs) $lhs)
; Returns: all defined rules

; ----------------------------------------------------------------------------
; Multiple Spaces Don't Interfere
; ----------------------------------------------------------------------------
!(bind-space &facts (new-space))
!(bind-space &rules (new-space))
!(add-atom &facts (sky blue))
!(add-atom &rules (infer $x))
!(match &facts $x $x)    ; Returns: [(sky blue)]
!(match &rules $x $x)    ; Returns: [(infer $x)]

; ----------------------------------------------------------------------------
; Anonymous Spaces (Local Variables)
; ----------------------------------------------------------------------------
; Use anonymous spaces without binding them to global names
!(let $workspace (new-space)
  (progn
    (add-atom $workspace (task 1 done))
    (add-atom $workspace (task 2 pending))
    (match $workspace (task $id done) $id)))
; Returns: [1]

; ----------------------------------------------------------------------------
; Unicode Support - λ-calculus and Logic
; ----------------------------------------------------------------------------
!(bind-space &λ (new-space))
!(add-atom &λ (λ x (+ x 1)))
!(add-atom &λ (∀x ∃y (relation x y)))
!(match &λ (λ $var $body) (lambda $var $body))
; Returns: [(lambda x (+ x 1))]

; ----------------------------------------------------------------------------
; Sequential Operations with let* (including spaces!)
; ----------------------------------------------------------------------------
; let* evaluates bindings sequentially (each can reference previous)
!(let* (($a 10)
        ($b (+ $a 5))
        ($c (* $b 2)))
  $c)
; Returns: [30]

; Advanced: Use let* with spaces (like mmverify pattern)
!(let* (($subst (new-space))
        (() (add-atom $subst (binding x 100)))
        (() (add-atom $subst (binding y 200))))
  (match $subst (binding x $val) $val))
; Returns: [100]

; ----------------------------------------------------------------------------
; Side Effects with println and progn
; ----------------------------------------------------------------------------
; progn executes sequentially, returns last value
!(progn
  (println "Building knowledge base...")
  (bind-space &demo (new-space))
  (add-atom &demo (demo works))
  (println "Done!")
  (match &demo $x $x))
; Prints: "Building knowledge base..." then "Done!"
; Returns: [(demo works)]

; ----------------------------------------------------------------------------
; Cleanup with delete-space
; ----------------------------------------------------------------------------
!(delete-space &facts)
!(match &facts $x $x)
; Returns: [(Error match: space 'facts' does not exist...)]

; Cannot delete &self
!(delete-space &self)
; Returns: [(Error delete-space: cannot delete the main 'self' space...)]

; ----------------------------------------------------------------------------
; Error Handling with Fuzzy Matching
; ----------------------------------------------------------------------------
!(match &slef $x $x)
; Returns: [(Error ... Did you mean: self? ...)]

!(match &Kb $x $x)
; Returns: [(Error ... Create it with: !(bind-space &Kb (new-space)) ...)]
