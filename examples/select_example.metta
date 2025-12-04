; select operator example - similar to Prolog's once/1
; Usage: (select k expr) returns the first k results from expr

; Define a function with multiple matching clauses
(= (color apple) red)
(= (color apple) green)
(= (color apple) yellow)

; Without select - returns all 3 colors
!(println ("All colors of apple:" (collapse (color apple))))

; With select 1 - returns only the first color (like Prolog's once/1)
!(println ("First color only:" (select 1 (color apple))))

; With select 2 - returns first two colors
!(println ("First two colors:" (collapse (select 2 (color apple)))))

; Select with arithmetic
!(println ("Arithmetic (returns one result, select 1 has no effect):" (select 1 (+ 2 3))))

; Select 0 - returns nothing
!(println ("Select 0 (returns nothing):" (collapse (select 0 (color apple)))))
