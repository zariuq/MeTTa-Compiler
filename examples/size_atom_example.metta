; size-atom: Count the number of atoms in an expression
; Usage: (size-atom expr) → count

; Example 1: Basic list counting
!(size-atom (a b c))
; Returns: [3]

!(size-atom (x))
; Returns: [1]

!(size-atom ())
; Returns: [0]

; Example 2: Counting clause literals (ATP)
!(size-atom ((lit p) (nlit q) (lit r)))
; Returns: [3]  - This clause has 3 literals

!(size-atom ((nlit s)))
; Returns: [1]  - Unit clause

; Example 3: Dynamic size calculation with let
!(let $clause1 ((lit a) (lit b))
  (size-atom $clause1))
; Returns: [2]

!(let $clause2 ((lit c))
  (size-atom $clause2))
; Returns: [1]

; Example 4: Check if clause is unit clause
!(let $clause ((nlit x))
  (if (== (size-atom $clause) 1)
    "Unit clause - highest priority!"
    "Multi-literal clause"))
; Returns: ["Unit clause - highest priority!"]

; Example 5: Counting function arguments
!(let $args (arg1 arg2 arg3 arg4)
  (size-atom $args))
; Returns: [4]

; Example 6: Nested structures (counts top-level only)
!(size-atom ((a b) (c d) (e f)))
; Returns: [3]  - Three top-level elements
