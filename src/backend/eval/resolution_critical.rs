// Critical resolution functions for performance
// These 5 functions are in the hot path of the DISCOUNT loop

use crate::backend::models::MettaValue;
use std::sync::Arc;
use std::collections::HashSet;

/// Check if clause1 subsumes clause2 (clause1 is more general)
/// A clause C1 subsumes C2 if every literal in C1 appears in C2
pub fn fast_subsumes(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 2 {
        return Some(MettaValue::Bool(false));
    }

    let clause1 = &args[0];
    let clause2 = &args[1];

    // Extract literals from clause1
    let mut lits1 = Vec::new();
    extract_literals(clause1, &mut lits1);

    // Extract literals from clause2
    let mut lits2 = Vec::new();
    extract_literals(clause2, &mut lits2);

    // Check if every literal in clause1 appears in clause2
    for lit1 in &lits1 {
        if !lits2.iter().any(|lit2| literals_equal(lit1, lit2)) {
            return Some(MettaValue::Bool(false));
        }
    }

    Some(MettaValue::Bool(true))
}

/// Check if a clause is subsumed by any clause in a list
pub fn fast_is_subsumed_by_any(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 2 {
        return Some(MettaValue::Bool(false));
    }

    let clause = &args[0];
    let mut current = &args[1];

    // Walk through the list
    loop {
        match current {
            MettaValue::Atom(s) if s == "Nil" => return Some(MettaValue::Bool(false)),
            MettaValue::SExpr(items) if items.len() == 3 => {
                if let MettaValue::Atom(cons) = &items[0] {
                    if cons == "Cons" {
                        // Check if this clause subsumes our clause
                        if let Some(MettaValue::Bool(true)) = fast_subsumes(&[items[1].clone(), clause.clone()]) {
                            return Some(MettaValue::Bool(true));
                        }
                        current = &items[2];
                        continue;
                    }
                }
                return Some(MettaValue::Bool(false));
            }
            _ => return Some(MettaValue::Bool(false)),
        }
    }
}

/// Core resolution operation - resolve two clauses
pub fn fast_resolve(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 2 {
        return Some(MettaValue::Atom("no-resolvent".to_string()));
    }

    let clause1 = &args[0];
    let clause2 = &args[1];

    // Extract literals
    let mut lits1 = Vec::new();
    extract_literals(clause1, &mut lits1);

    let mut lits2 = Vec::new();
    extract_literals(clause2, &mut lits2);

    // Try to find complementary literals
    for (i, lit1) in lits1.iter().enumerate() {
        for (j, lit2) in lits2.iter().enumerate() {
            if are_complementary_lits(lit1, lit2) {
                // Found complementary pair - create resolvent
                let mut resolvent_lits = Vec::new();

                // Add all literals from clause1 except lit1
                for (k, l) in lits1.iter().enumerate() {
                    if k != i {
                        resolvent_lits.push(l.clone());
                    }
                }

                // Add all literals from clause2 except lit2
                for (k, l) in lits2.iter().enumerate() {
                    if k != j {
                        // Only add if not already present (dedupe)
                        if !resolvent_lits.iter().any(|existing| literals_equal(existing, l)) {
                            resolvent_lits.push(l.clone());
                        }
                    }
                }

                // Check for tautology
                if is_tautological(&resolvent_lits) {
                    return Some(MettaValue::Atom("no-resolvent".to_string()));
                }

                // Build result clause
                return Some(build_clause_from_literals(&resolvent_lits));
            }
        }
    }

    Some(MettaValue::Atom("no-resolvent".to_string()))
}

/// Check if a clause is a tautology (contains P and ¬P)
pub fn fast_is_tautology(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 1 {
        return Some(MettaValue::Bool(false));
    }

    let mut literals = Vec::new();
    extract_literals(&args[0], &mut literals);

    Some(MettaValue::Bool(is_tautological(&literals)))
}

/// Remove duplicate clauses from a list
pub fn fast_dedupe(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 1 {
        return Some(MettaValue::Atom("Nil".to_string()));
    }

    let mut seen = HashSet::new();
    let mut result = Vec::new();

    let mut current = &args[0];
    loop {
        match current {
            MettaValue::Atom(s) if s == "Nil" => break,
            MettaValue::SExpr(items) if items.len() == 3 => {
                if let MettaValue::Atom(cons) = &items[0] {
                    if cons == "Cons" {
                        let clause_str = format!("{:?}", items[1]);
                        if !seen.contains(&clause_str) {
                            seen.insert(clause_str);
                            result.push(items[1].clone());
                        }
                        current = &items[2];
                        continue;
                    }
                }
                break;
            }
            _ => break,
        }
    }

    // Build result list
    let mut list = MettaValue::Atom("Nil".to_string());
    for clause in result.into_iter().rev() {
        list = MettaValue::SExpr(vec![
            MettaValue::Atom("Cons".to_string()),
            clause,
            list,
        ]);
    }

    Some(list)
}

// Helper functions

fn extract_literals(clause: &MettaValue, literals: &mut Vec<MettaValue>) {
    match clause {
        MettaValue::Atom(s) if s == "Nil" => {},
        MettaValue::SExpr(items) if items.len() == 3 => {
            if let MettaValue::Atom(cons) = &items[0] {
                if cons == "Cons" {
                    literals.push(items[1].clone());
                    extract_literals(&items[2], literals);
                }
            }
        }
        _ => {}
    }
}

fn literals_equal(lit1: &MettaValue, lit2: &MettaValue) -> bool {
    format!("{:?}", lit1) == format!("{:?}", lit2)
}

fn are_complementary_lits(lit1: &MettaValue, lit2: &MettaValue) -> bool {
    match (lit1, lit2) {
        (MettaValue::SExpr(items1), MettaValue::SExpr(items2))
            if items1.len() == 2 && items2.len() == 2 => {
            match (&items1[0], &items2[0]) {
                (MettaValue::Atom(s1), MettaValue::Atom(s2)) => {
                    if s1 == "lit" && s2 == "nlit" {
                        return items1[1] == items2[1];
                    }
                    if s1 == "nlit" && s2 == "lit" {
                        return items1[1] == items2[1];
                    }
                }
                _ => {}
            }
        }
        _ => {}
    }
    false
}

fn is_tautological(literals: &[MettaValue]) -> bool {
    for i in 0..literals.len() {
        for j in (i+1)..literals.len() {
            if are_complementary_lits(&literals[i], &literals[j]) {
                return true;
            }
        }
    }
    false
}

fn build_clause_from_literals(literals: &[MettaValue]) -> MettaValue {
    let mut clause = MettaValue::Atom("Nil".to_string());
    for lit in literals.iter().rev() {
        clause = MettaValue::SExpr(vec![
            MettaValue::Atom("Cons".to_string()),
            lit.clone(),
            clause,
        ]);
    }
    clause
}