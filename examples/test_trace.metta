;; Test actual non-determinism with multiple rules of same specificity
;; Both rules match (multi 1) with same specificity (both have exact match on 1)

(= (multi 1) result-a)
(= (multi 1) result-b)

!(multi 1)
