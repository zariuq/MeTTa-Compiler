; Fixed version for MeTTa-Compiler
; Uses `let` to force evaluation of nested calls

; Test map-atom
!(map-atom (1 2 3 4) $v (+ $v 1))

; Test filter-atom
!(filter-atom (1 2 3 4 5 6) $v (> $v 3))

; Test foldl-atom
!(foldl-atom (1 2 3 4) 0 $acc $x (+ $acc $x))

; Test composition - FIXED with let
!(let $mapped (map-atom (1 2 3 4 5) $v (* $v 2))
   (let $filtered (filter-atom $mapped $v (> $v 4))
      (foldl-atom $filtered 0 $acc $x (+ $acc $x))))
