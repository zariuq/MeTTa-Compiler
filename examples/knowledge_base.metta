; ============================================================================
; Knowledge Base Demo - Family Tree
; ============================================================================

!(println "Building family tree knowledge base...")

; Create a dedicated space for family facts
!(bind-space &family (new-space))

; Add family relationships
!(add-atom &family (parent alice bob))
!(add-atom &family (parent alice charlie))
!(add-atom &family (parent bob david))
!(add-atom &family (parent bob emma))
!(add-atom &family (parent charlie frank))

; Query: Who are Alice's children?
!(println "Alice's children:")
!(match &family (parent alice $child) $child)

; Query: Who are Bob's children?
!(println "Bob's children:")
!(match &family (parent bob $child) $child)

; Query: All parents
!(println "All parents:")
!(match &family (parent $p $c) $p)

; Use let* to compute grandchildren
!(println "Alice's grandchildren:")
!(let* (($children (match &family (parent alice $c) $c))
        ($grandchildren (match &family (parent $children $gc) $gc)))
       $grandchildren)

