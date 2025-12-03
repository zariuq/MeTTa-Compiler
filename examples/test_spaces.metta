; Test bind-space and explicit space creation

; Test 1: Create spaces and add atoms
!(bind-space &kb)
!(add-atom &kb (clause p q))
!(add-atom &kb (clause r s))
!(add-atom &kb (clause p t))

; Test 2: Match in the space
!(match &kb (clause $x $y) (found $x $y))

; Test 3: Create separate spaces that don't interfere
!(bind-space &positive)
!(bind-space &negative)

!(add-atom &positive (lit p (clause1)))
!(add-atom &negative (lit (neg p) (clause2)))

!(match &positive (lit $l $c) (positive-result $l $c))
!(match &negative (lit $l $c) (negative-result $l $c))

; Test 4: &self still works for rules (no bind-space needed)
(= (test-rule $x) (result $x))
!(match &self (test-rule $y) $y)

; Test 5: Create a facts space and add multiple facts
!(bind-space &facts)
!(add-atom &facts (person alice 25))
!(add-atom &facts (person bob 30))
!(add-atom &facts (person charlie 35))

!(match &facts (person $name $age) $name)
