; Demonstration of collapse and superpose operators
; These are inverse operations for working with non-deterministic evaluation

; Define a non-deterministic function with multiple matching clauses
(= (color) red)
(= (color) green)
(= (color) blue)

; Without collapse - returns 3 separate results
!(println ("Without collapse (3 separate results):" (color)))

; Collapse: MANY results -> ONE S-expression
!(println ("With collapse (one list containing all results):" (collapse (color))))

; Superpose: ONE S-expression -> MANY results
!(println ("Superpose (explode list into multiple results):" (superpose (A B C D))))

; Roundtrip demonstration: 
!(println ("Roundtrip test: collapse(superpose(x)) = x" (collapse (superpose (X Y Z)))))

; Combining with select operator
!(println ("Select first 2 from superpose:" (collapse (select 2 (superpose (apple banana cherry date))))))

; Practical example: collecting from non-deterministic choices
(= (fruit) apple)
(= (fruit) banana)
(= (fruit) cherry)

!(println ("Collect all fruits into a list:" (collapse (fruit))))
