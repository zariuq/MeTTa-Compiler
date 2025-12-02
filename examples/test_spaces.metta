; Test PeTTa-style auto-creating spaces

; Test 1: Add atoms to a space
!(add-atom &kb (clause p q))
!(add-atom &kb (clause r s))
!(add-atom &kb (clause p t))

; Test 2: Match in the space
!(match &kb (clause $x $y) (found $x $y))

; Test 3: Separate spaces don't interfere
!(add-atom &positive (lit p (clause1)))
!(add-atom &negative (lit (neg p) (clause2)))

!(match &positive (lit $l $c) (positive-result $l $c))
!(match &negative (lit $l $c) (negative-result $l $c))

; Test 4: &self still works for rules
(= (test-rule $x) (result $x))
!(match &self (test-rule $y) $y)

; Test 5: Add multiple facts and match
!(add-atom &facts (person alice 25))
!(add-atom &facts (person bob 30))
!(add-atom &facts (person charlie 35))

!(match &facts (person $name $age) $name)
