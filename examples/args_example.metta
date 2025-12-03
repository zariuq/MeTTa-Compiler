; Command-Line Arguments Example
; Run: mettatron examples/args_example.metta 100 timeout=30 strategy=discount
; Or with dash args: mettatron examples/args_example.metta --args --verbose -n=5 5 timeout=30 strategy=unit

!(println "=== Command-Line Arguments Demo ===")

!(println "\nAll arguments:")
!(get-args)

!(println "\nFirst positional argument:")
!(get-arg 0)

!(println "\nSecond positional argument:")
!(get-arg 1)

!(println "\nGet timeout value (from key=value):")
!(get-arg "timeout")

!(println "\nGet strategy value:")
!(get-arg "strategy")
