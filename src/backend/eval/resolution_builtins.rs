// Fast built-in implementations of resolution prover functions
// This demonstrates the "fast Rust library" approach

use crate::backend::models::MettaValue;
use std::sync::Arc;

/// Fast member check - O(n) list traversal without pattern matching
pub fn fast_member(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 2 {
        return Some(MettaValue::Error(
            format!("member? requires exactly 2 arguments, got {}", args.len()),
            Arc::new(MettaValue::Nil),
        ));
    }

    let needle = &args[0];
    let mut current = &args[1];

    loop {
        match current {
            MettaValue::Atom(s) if s == "Nil" => return Some(MettaValue::Bool(false)),
            MettaValue::SExpr(items) => {
                // Check if it's a Cons structure
                if items.len() == 3 {
                    if let MettaValue::Atom(cons) = &items[0] {
                        if cons == "Cons" {
                            // Compare head
                            if needle == &items[1] {
                                return Some(MettaValue::Bool(true));
                            }
                            // Continue with tail
                            current = &items[2];
                            continue;
                        }
                    }
                }
                // Not a proper Cons structure
                return Some(MettaValue::Bool(false));
            }
            _ => return Some(MettaValue::Bool(false)),
        }
    }
}

/// Fast list length - O(n) without pattern matching
pub fn fast_list_length(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 1 {
        return Some(MettaValue::Error(
            format!("list-length requires exactly 1 argument, got {}", args.len()),
            Arc::new(MettaValue::Nil),
        ));
    }

    let mut count = 0i64;
    let mut current = &args[0];

    loop {
        match current {
            MettaValue::Atom(s) if s == "Nil" => return Some(MettaValue::Long(count)),
            MettaValue::SExpr(items) => {
                // Check if it's a Cons structure
                if items.len() == 3 {
                    if let MettaValue::Atom(cons) = &items[0] {
                        if cons == "Cons" {
                            count += 1;
                            current = &items[2]; // tail
                            continue;
                        }
                    }
                }
                // Not a proper list
                return Some(MettaValue::Error(
                    "Invalid list structure".to_string(),
                    Arc::new(args[0].clone()),
                ));
            }
            _ => return Some(MettaValue::Error(
                "Invalid list structure".to_string(),
                Arc::new(args[0].clone()),
            )),
        }
    }
}

/// Fast append - O(n) where n is length of first list
pub fn fast_append_sets(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 2 {
        return Some(MettaValue::Error(
            format!("append-sets requires exactly 2 arguments, got {}", args.len()),
            Arc::new(MettaValue::Nil),
        ));
    }

    // Helper to build the result
    fn append_recursive(first: &MettaValue, second: &MettaValue) -> MettaValue {
        match first {
            MettaValue::Atom(s) if s == "Nil" => second.clone(),
            MettaValue::SExpr(items) if items.len() == 3 => {
                if let MettaValue::Atom(cons) = &items[0] {
                    if cons == "Cons" {
                        // Cons head (append tail second)
                        let new_tail = append_recursive(&items[2], second);
                        return MettaValue::SExpr(vec![
                            MettaValue::Atom("Cons".to_string()),
                            items[1].clone(),
                            new_tail,
                        ]);
                    }
                }
                // Not a Cons, return as-is
                first.clone()
            }
            _ => first.clone(),
        }
    }

    Some(append_recursive(&args[0], &args[1]))
}

/// Check if a list is empty - O(1)
pub fn fast_is_empty(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 1 {
        return Some(MettaValue::Error(
            format!("is-empty requires exactly 1 argument, got {}", args.len()),
            Arc::new(MettaValue::Nil),
        ));
    }

    match &args[0] {
        MettaValue::Atom(s) if s == "Nil" => Some(MettaValue::Bool(true)),
        _ => Some(MettaValue::Bool(false)),
    }
}

/// Extract head of list (car) - O(1)
pub fn fast_car(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 1 {
        return Some(MettaValue::Error(
            format!("car requires exactly 1 argument, got {}", args.len()),
            Arc::new(MettaValue::Nil),
        ));
    }

    match &args[0] {
        MettaValue::Atom(s) if s == "Nil" => Some(MettaValue::Atom("Nil".to_string())),
        MettaValue::SExpr(items) if items.len() == 3 => {
            if let MettaValue::Atom(cons) = &items[0] {
                if cons == "Cons" {
                    return Some(items[1].clone());
                }
            }
            Some(MettaValue::Atom("Nil".to_string()))
        }
        _ => Some(MettaValue::Atom("Nil".to_string())),
    }
}

/// Extract tail of list (cdr) - O(1)
pub fn fast_cdr(args: &[MettaValue]) -> Option<MettaValue> {
    if args.len() != 1 {
        return Some(MettaValue::Error(
            format!("cdr requires exactly 1 argument, got {}", args.len()),
            Arc::new(MettaValue::Nil),
        ));
    }

    match &args[0] {
        MettaValue::Atom(s) if s == "Nil" => Some(MettaValue::Atom("Nil".to_string())),
        MettaValue::SExpr(items) if items.len() == 3 => {
            if let MettaValue::Atom(cons) = &items[0] {
                if cons == "Cons" {
                    return Some(items[2].clone());
                }
            }
            Some(MettaValue::Atom("Nil".to_string()))
        }
        _ => Some(MettaValue::Atom("Nil".to_string())),
    }
}