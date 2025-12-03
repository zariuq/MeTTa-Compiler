; Import Example - demonstrates importing definitions from other files
; Run: mettatron examples/import_example.metta

!(println "Loading math library...")
!(import! &self "./math_lib.metta")

; Use imported definitions
!(println "Testing imported functions:")
!(double 5)
!(square 4)
!(cube 3)
