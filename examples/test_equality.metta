; Test equality in detail

; Numeric equality
!(== 5 5)
!(== 5 6)

; Symbolic equality - does this work?
!(== a a)
!(== a b)

; String equality
!(== "hello" "hello")
!(== "hello" "world")

; Pattern matching vs equality
(= (test-equal $x $x) same)
(= (test-equal $x $y) different)

!(test-equal a a)
!(test-equal a b)
