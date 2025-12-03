; remove-atom: Remove atoms from spaces
; Usage: (remove-atom &space-name atom)

; Example 1: Remove from main space (&self)
!(add-atom &self (person alice))
!(add-atom &self (person bob))
!(add-atom &self (person charlie))

; View all people
!(match &self (person $name) $name)
; Returns: [alice, bob, charlie]

; Remove Bob
!(remove-atom &self (person bob))

; View remaining people
!(match &self (person $name) $name)
; Returns: [alice, charlie]

; Example 2: Remove from named space
; First, create the database space
!(bind-space &database)

!(add-atom &database (fact (temperature 20)))
!(add-atom &database (fact (humidity 65)))
!(add-atom &database (fact (pressure 1013)))

; View all facts
!(match &database (fact $f) $f)
; Returns: [(temperature 20), (humidity 65), (pressure 1013)]

; Remove outdated temperature reading
!(remove-atom &database (fact (temperature 20)))

; View remaining facts
!(match &database (fact $f) $f)
; Returns: [(humidity 65), (pressure 1013)]

; Example 3: ATP use case - clause database cleanup
; First, create the clauses space
!(bind-space &clauses)

!(add-atom &clauses (clause ((lit p) ((nlit q) ()))))
!(add-atom &clauses (clause ((lit p) ())))  ; Subsumes the first one!

; Before cleanup
!(match &clauses (clause $lits) $lits)
; Returns: [((lit p) Nil), ((lit p) ((nlit q) Nil))]

; Remove subsumed clause
!(remove-atom &clauses (clause ((lit p) ((nlit q) ()))))

; After cleanup
!(match &clauses (clause $lits) $lits)
; Returns: [((lit p) Nil)]
