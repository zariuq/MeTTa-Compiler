; Correct case syntax for MeTTa-Compiler

; Syntax: (case VALUE ((pattern1 result1) (pattern2 result2) ...))

!(case Nil
   ((Nil empty)
    ((Cons $h $t) non-empty)))

!(case (Cons a Nil)
   ((Nil empty)
    ((Cons $h $t) (has-head $h))))

!(case (Cons a (Cons b Nil))
   ((Nil empty)
    ((Cons $h Nil) (single $h))
    ((Cons $h $t) (multiple $h))))

; Use in a function
(= (list-type Nil) empty)
(= (list-type (Cons $h Nil)) single)
(= (list-type (Cons $h $t)) multiple)

!(list-type Nil)
!(list-type (Cons a Nil))
!(list-type (Cons a (Cons b Nil)))
