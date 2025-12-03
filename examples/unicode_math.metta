; ============================================================================
; Unicode Math - λ-calculus and Logic
; ============================================================================

!(println "Unicode mathematical notation demo...")

; --- Lambda Calculus ---
!(bind-space &λ (new-space))
!(add-atom &λ (λ x (+ x 1)))
!(add-atom &λ (λ y (* y y)))
!(add-atom &λ (λ f (λ x (f (f x)))))

!(println "Lambda terms:")
!(match &λ (λ $var $body) (term $var $body))

; --- First-Order Logic ---
!(bind-space &logic (new-space))
!(add-atom &logic (∀x (P x)))
!(add-atom &logic (∃y (Q y)))
!(add-atom &logic (∀x ∃y (R x y)))
!(add-atom &logic (∀x (P x → Q x)))

!(println "Universal quantification:")
!(match &logic (∀x $formula) $formula)

!(println "Existential quantification:")
!(match &logic (∃y $formula) $formula)

; --- Mathematical Symbols ---
!(bind-space &math (new-space))
!(add-atom &math (π 3.14159))
!(add-atom &math (α ≤ β))
!(add-atom &math (γ ∈ Γ))
!(add-atom &math (∑ i 1 n))

!(println "Mathematical constants and relations:")
!(match &math $x $x)

; --- Set Theory ---
!(bind-space &sets (new-space))
!(add-atom &sets (A ∪ B))
!(add-atom &sets (A ∩ B))
!(add-atom &sets (A ⊆ B))
!(add-atom &sets (x ∈ A))

!(println "Set operations:")
!(match &sets ($A ∪ $B) (union $A $B))
!(match &sets ($A ∩ $B) (intersection $A $B))

!(println "Unicode math demo complete! ✓")
