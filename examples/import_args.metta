!(import! &self "./math_lib.metta")
!(let $file (get-arg 0) (import! &self $file))

; Use imported definitions
!(println "Testing imported functions:")
!(double 5)
!(square 4)
!(cube 3)

