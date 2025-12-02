; max-atom: Find the maximum value in a list of numbers
; Usage: (max-atom list) → maximum

; Example 1: Find maximum in a list
!(max-atom (1 5 3 9 2))
; Returns: [9]

!(max-atom (42))
; Returns: [42]

; Example 2: Works with negative numbers
!(max-atom (-5 -2 -10 -1))
; Returns: [-1]

; Example 3: Works with floats
!(max-atom (1.5 2.7 0.9 3.2))
; Returns: [3.2]

; Example 4: Mixed integers and floats
!(max-atom (1 2.5 3 4.1))
; Returns: [4.1]

; Example 5: ATP use case - Find maximum clause ID
!(max-atom (1 5 3 9))
; Returns: [9]

; Example 6: Find maximum depth in proof search
!(max-atom (5 3 8))
; Returns: [8]

; Example 7: Weight-based term ordering
!(let $weights (5 3 7 2 9)
  (if (> (max-atom $weights) 5)
    "Heavy term - order carefully"
    "Light term - safe to order"))
; Returns: ["Heavy term - order carefully"]
