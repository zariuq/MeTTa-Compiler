use crate::backend::environment::Environment;
use crate::backend::models::{EvalResult, MettaValue};
use std::sync::Arc;

use super::{apply_bindings, eval, pattern_match};

/// Format a MettaValue for printing (without outer brackets)
fn format_for_print(value: &MettaValue) -> String {
    match value {
        MettaValue::Atom(s) => s.clone(),
        MettaValue::Bool(b) => b.to_string(),
        MettaValue::Long(n) => n.to_string(),
        MettaValue::Float(f) => f.to_string(),
        MettaValue::String(s) => s.clone(), // Don't add quotes for println
        MettaValue::Nil => "()".to_string(),
        MettaValue::Error(msg, details) => {
            format!("(Error {} {})", msg, format_for_print(details))
        }
        MettaValue::Type(t) => format!("Type({})", format_for_print(t)),
        MettaValue::Space(uuid) => format!("GroundingSpace-{}", &uuid[..8]),
        MettaValue::SExpr(items) => {
            let formatted: Vec<String> = items.iter().map(format_for_print).collect();
            format!("({})", formatted.join(" "))
        }
    }
}

/// Evaluate println: (println! arg)
/// Prints the argument to stdout and returns ()
/// Following HE convention: println! takes exactly 1 argument
pub(super) fn eval_println(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    // Strict validation: exactly 1 argument (HE compatible)
    if args.len() != 1 {
        let err = MettaValue::Error(
            format!(
                "println! requires exactly 1 argument, got {}. Usage: (println! value)",
                args.len()
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    // Evaluate the first argument
    let (results, new_env) = eval(args[0].clone(), env);

    if results.is_empty() {
        return (vec![MettaValue::Nil], new_env);
    }

    // Print each result
    for result in &results {
        println!("{}", format_for_print(result));
    }

    // Return () (allows use in let bindings, fire-and-forget semantics)
    (vec![MettaValue::Nil], new_env)
}

/// Generate helpful message for pattern mismatch in let bindings
fn pattern_mismatch_suggestion(pattern: &MettaValue, value: &MettaValue) -> String {
    let pattern_arity = match pattern {
        MettaValue::SExpr(items) => items.len(),
        _ => 1,
    };
    let value_arity = match value {
        MettaValue::SExpr(items) => items.len(),
        _ => 1,
    };

    // Check for arity mismatch
    if pattern_arity != value_arity {
        return format!(
            "Hint: pattern has {} element(s) but value has {}. Adjust pattern to match value structure.",
            pattern_arity, value_arity
        );
    }

    // Check for structure mismatch (different head atoms)
    if let (MettaValue::SExpr(p_items), MettaValue::SExpr(v_items)) = (pattern, value) {
        if let (Some(MettaValue::Atom(p_head)), Some(MettaValue::Atom(v_head))) =
            (p_items.first(), v_items.first())
        {
            if p_head != v_head {
                return format!(
                    "Hint: pattern head '{}' doesn't match value head '{}'.",
                    p_head, v_head
                );
            }
        }
    }

    // Check for literal mismatch inside structures
    if let (MettaValue::SExpr(p_items), MettaValue::SExpr(v_items)) = (pattern, value) {
        for (i, (p, v)) in p_items.iter().zip(v_items.iter()).enumerate() {
            // Skip if pattern is a variable (starts with $, &, or ')
            if let MettaValue::Atom(name) = p {
                if name.starts_with('$')
                    || name.starts_with('&')
                    || name.starts_with('\'')
                    || name == "_"
                {
                    continue;
                }
            }
            // Check for literal mismatch
            if p != v && !matches!(p, MettaValue::SExpr(_)) {
                return format!(
                    "Hint: element at position {} doesn't match - pattern has {:?} but value has {:?}.",
                    i, p, v
                );
            }
        }
    }

    // Default hint
    "Hint: pattern structure doesn't match value. Check that variable names align with value positions.".to_string()
}

/// Evaluate let binding: (let pattern value body)
/// Evaluates value, binds it to pattern, and evaluates body with those bindings
/// Supports both simple variable binding and pattern matching:
///   - (let $x 42 body) - simple binding
///   - (let ($a $b) (tuple 1 2) body) - destructuring pattern
pub(super) fn eval_let(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.len() < 3 {
        let got = args.len();
        let err = MettaValue::Error(
            format!(
                "let requires exactly 3 arguments, got {}. Usage: (let pattern value body)",
                got
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    let pattern = &args[0];
    let value_expr = &args[1];
    let body = &args[2];

    // Evaluate the value expression first
    let (value_results, value_env) = eval(value_expr.clone(), env);

    // Handle nondeterminism: if value evaluates to multiple results, try each one
    let mut all_results = Vec::new();

    for value in value_results {
        // Try to match the pattern against the value
        if let Some(bindings) = pattern_match(pattern, &value) {
            // Apply bindings to the body and evaluate it
            let instantiated_body = apply_bindings(body, &bindings);
            let (body_results, _) = eval(instantiated_body, value_env.clone());
            all_results.extend(body_results);
        } else {
            // Pattern match failed - provide helpful suggestion
            let suggestion = pattern_mismatch_suggestion(pattern, &value);
            let err = MettaValue::Error(
                format!(
                    "let pattern {} does not match value {}. {}",
                    super::friendly_value_repr(pattern),
                    super::friendly_value_repr(&value),
                    suggestion
                ),
                Arc::new(MettaValue::SExpr(args.to_vec())),
            );
            all_results.push(err);
        }
    }

    (all_results, value_env)
}

/// Evaluate let*: (let* ((pattern1 value1) (pattern2 value2) ...) body)
/// Sequential let bindings where each binding can reference previous ones
/// Example: (let* (($a 1) ($b (+ $a 1))) (+ $a $b)) → 3
pub(super) fn eval_let_star(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.len() < 2 {
        let got = args.len();
        let err = MettaValue::Error(
            format!(
                "let* requires at least 2 arguments, got {}. Usage: (let* ((pattern value) ...) body)",
                got
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    let bindings_expr = &args[0];
    let body = &args[1];

    // Extract bindings list
    let bindings = match bindings_expr {
        MettaValue::SExpr(items) => items,
        _ => {
            let err = MettaValue::Error(
                format!(
                    "let* bindings must be a list, got: {}",
                    super::friendly_value_repr(bindings_expr)
                ),
                Arc::new(MettaValue::SExpr(args.to_vec())),
            );
            return (vec![err], env);
        }
    };

    // Process bindings sequentially (left to right)
    let current_env = env;
    let mut current_body = body.clone();

    for binding in bindings.iter().rev() {
        // Each binding is (pattern value)
        let (pattern, value_expr) = match binding {
            MettaValue::SExpr(items) if items.len() == 2 => (&items[0], &items[1]),
            _ => {
                let err = MettaValue::Error(
                    format!(
                        "let* binding must be (pattern value), got: {}",
                        super::friendly_value_repr(binding)
                    ),
                    Arc::new(MettaValue::SExpr(args.to_vec())),
                );
                return (vec![err], current_env);
            }
        };

        // Create nested let: (let pattern value current_body)
        current_body = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            pattern.clone(),
            value_expr.clone(),
            current_body,
        ]);
    }

    // Evaluate the nested structure
    eval(current_body, current_env)
}

/// Evaluate progn: (progn expr1 expr2 ... exprN)
/// Evaluates all expressions sequentially and returns the result of the LAST one
/// Example: (progn (println "step1") (println "step2") 42) → prints both, returns 42
pub(super) fn eval_progn(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.is_empty() {
        return (vec![MettaValue::Nil], env);
    }

    let mut current_env = env;
    let mut last_results = vec![MettaValue::Nil];

    // Evaluate each expression in sequence
    for expr in args {
        let (results, new_env) = eval(expr.clone(), current_env);
        current_env = new_env;
        last_results = results;
    }

    (last_results, current_env)
}

/// Evaluate prog1: (prog1 expr1 expr2 ... exprN)
/// Evaluates all expressions sequentially and returns the result of the FIRST one
/// Example: (prog1 42 (println "side effect")) → prints, returns 42
pub(super) fn eval_prog1(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.is_empty() {
        return (vec![MettaValue::Nil], env);
    }

    // Evaluate first expression and save result
    let (first_results, mut current_env) = eval(args[0].clone(), env);

    // Evaluate remaining expressions for side effects
    for expr in &args[1..] {
        let (_results, new_env) = eval(expr.clone(), current_env);
        current_env = new_env;
    }

    (first_results, current_env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_let_simple_binding() {
        let env = Environment::new();

        // (let $x 42 $x)
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$x".to_string()),
            MettaValue::Long(42),
            MettaValue::Atom("$x".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(42));
    }

    #[test]
    fn test_let_with_expression() {
        let env = Environment::new();

        // (let $y (+ 10 5) (* $y 2))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$y".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(10),
                MettaValue::Long(5),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("*".to_string()),
                MettaValue::Atom("$y".to_string()),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(30));
    }

    #[test]
    fn test_let_with_pattern_matching() {
        let env = Environment::new();

        // (let (tuple $a $b) (tuple 1 2) (+ $a $b))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("tuple".to_string()),
                MettaValue::Atom("$a".to_string()),
                MettaValue::Atom("$b".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("tuple".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Atom("$a".to_string()),
                MettaValue::Atom("$b".to_string()),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(3));
    }

    #[test]
    fn test_let_nested() {
        let env = Environment::new();

        // (let $z 3 (let $w 4 (+ $z $w)))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$z".to_string()),
            MettaValue::Long(3),
            MettaValue::SExpr(vec![
                MettaValue::Atom("let".to_string()),
                MettaValue::Atom("$w".to_string()),
                MettaValue::Long(4),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("+".to_string()),
                    MettaValue::Atom("$z".to_string()),
                    MettaValue::Atom("$w".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(7));
    }

    #[test]
    fn test_let_with_if() {
        let env = Environment::new();

        // (let $base 10 (if (> $base 5) (* $base 2) $base))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$base".to_string()),
            MettaValue::Long(10),
            MettaValue::SExpr(vec![
                MettaValue::Atom("if".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom(">".to_string()),
                    MettaValue::Atom("$base".to_string()),
                    MettaValue::Long(5),
                ]),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("*".to_string()),
                    MettaValue::Atom("$base".to_string()),
                    MettaValue::Long(2),
                ]),
                MettaValue::Atom("$base".to_string()),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(20));
    }

    #[test]
    fn test_let_pattern_mismatch() {
        let env = Environment::new();

        // (let (foo $x) (bar 42) $x) - pattern mismatch should error
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("foo".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("bar".to_string()),
                MettaValue::Long(42),
            ]),
            MettaValue::Atom("$x".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("does not match"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_let_with_wildcard_pattern() {
        let env = Environment::new();

        // (let _ 42 "ignored")
        // Wildcard should match anything but not bind
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("_".to_string()),
            MettaValue::Long(42),
            MettaValue::String("ignored".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("ignored".to_string()));
    }

    #[test]
    fn test_let_with_complex_pattern_structures() {
        let env = Environment::new();

        // (let (nested (inner $x $y) $z) (nested (inner 1 2) 3) (+ $x (+ $y $z)))
        let complex_pattern = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("nested".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("inner".to_string()),
                    MettaValue::Atom("$x".to_string()),
                    MettaValue::Atom("$y".to_string()),
                ]),
                MettaValue::Atom("$z".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("nested".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("inner".to_string()),
                    MettaValue::Long(1),
                    MettaValue::Long(2),
                ]),
                MettaValue::Long(3),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("+".to_string()),
                    MettaValue::Atom("$y".to_string()),
                    MettaValue::Atom("$z".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(complex_pattern, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(6)); // 1 + (2 + 3)
    }

    #[test]
    fn test_let_with_variable_consistency() {
        let env = Environment::new();

        // Test that same variable in pattern must match same value
        // (let (same $x $x) (same 5 5) (* $x 2))
        let consistent_vars = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("same".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("same".to_string()),
                MettaValue::Long(5),
                MettaValue::Long(5),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("*".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(consistent_vars, env.clone());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(10)); // 5 * 2

        // Test inconsistent variables should fail
        // (let (same $x $x) (same 5 7) (* $x 2))
        let inconsistent_vars = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("same".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("same".to_string()),
                MettaValue::Long(5),
                MettaValue::Long(7), // Different value - should fail
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("*".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(inconsistent_vars, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("does not match"));
            }
            _ => panic!("Expected pattern match error"),
        }
    }

    #[test]
    fn test_let_with_different_variable_types() {
        let env = Environment::new();

        // Test different variable prefixes: $, &, '
        let mixed_vars = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("triple".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Atom("&y".to_string()),
                MettaValue::Atom("'z".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("triple".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
                MettaValue::Long(3),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("+".to_string()),
                    MettaValue::Atom("&y".to_string()),
                    MettaValue::Atom("'z".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(mixed_vars, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(6)); // 1 + (2 + 3)
    }

    #[test]
    fn test_let_missing_arguments() {
        let env = Environment::new();

        // Test let with only 2 arguments
        let let_two_args = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$x".to_string()),
            MettaValue::Long(42),
        ]);
        let (results, _) = eval(let_two_args, env.clone());
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("let"), "Expected 'let' in: {}", msg);
                assert!(
                    msg.contains("3 arguments"),
                    "Expected '3 arguments' in: {}",
                    msg
                );
                assert!(msg.contains("got 2"), "Expected 'got 2' in: {}", msg);
                assert!(msg.contains("Usage:"), "Expected 'Usage:' in: {}", msg);
            }
            _ => panic!("Expected error for missing arguments"),
        }

        // Test let with only 1 argument
        let let_one_arg = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$x".to_string()),
        ]);
        let (results, _) = eval(let_one_arg, env.clone());
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("let"), "Expected 'let' in: {}", msg);
                assert!(
                    msg.contains("3 arguments"),
                    "Expected '3 arguments' in: {}",
                    msg
                );
                assert!(msg.contains("got 1"), "Expected 'got 1' in: {}", msg);
            }
            _ => panic!("Expected error for missing arguments"),
        }

        // Test let with no arguments
        let let_no_args = MettaValue::SExpr(vec![MettaValue::Atom("let".to_string())]);
        let (results, _) = eval(let_no_args, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("let"), "Expected 'let' in: {}", msg);
                assert!(
                    msg.contains("3 arguments"),
                    "Expected '3 arguments' in: {}",
                    msg
                );
                assert!(msg.contains("got 0"), "Expected 'got 0' in: {}", msg);
            }
            _ => panic!("Expected error for missing arguments"),
        }
    }

    #[test]
    fn test_let_with_evaluated_value_expression() {
        let env = Environment::new();

        // Test let where value needs evaluation
        // (let $result (+ (* 3 4) 5) (if (> $result 10) "big" "small"))
        let eval_value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$result".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("*".to_string()),
                    MettaValue::Long(3),
                    MettaValue::Long(4),
                ]),
                MettaValue::Long(5),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("if".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom(">".to_string()),
                    MettaValue::Atom("$result".to_string()),
                    MettaValue::Long(10),
                ]),
                MettaValue::String("big".to_string()),
                MettaValue::String("small".to_string()),
            ]),
        ]);

        let (results, _) = eval(eval_value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("big".to_string())); // 17 > 10
    }

    #[test]
    fn test_let_with_error_in_value() {
        let env = Environment::new();

        // Test let where value expression produces error
        // (let $x (error "value-error" nil) $x)
        let error_value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$x".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("error".to_string()),
                MettaValue::String("value-error".to_string()),
                MettaValue::Nil,
            ]),
            MettaValue::Atom("$x".to_string()),
        ]);

        let (results, _) = eval(error_value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert_eq!(msg, "value-error");
            }
            _ => panic!("Expected error to be bound and returned"),
        }
    }

    // === Tests for "Did You Mean" pattern mismatch suggestions ===

    #[test]
    fn test_pattern_mismatch_arity_hint() {
        let env = Environment::new();

        // (let ($a $b) (tuple 1 2 3) ...) - pattern has 2 elements, value has 3
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("$a".to_string()),
                MettaValue::Atom("$b".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("tuple".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
                MettaValue::Long(3),
            ]),
            MettaValue::Atom("$a".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("Hint"), "Expected 'Hint' in: {}", msg);
                assert!(
                    msg.contains("2 element"),
                    "Expected arity mismatch hint in: {}",
                    msg
                );
            }
            _ => panic!("Expected error with pattern mismatch hint"),
        }
    }

    #[test]
    fn test_pattern_mismatch_head_hint() {
        let env = Environment::new();

        // (let (foo $x) (bar 42) $x) - head atoms don't match
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("foo".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("bar".to_string()),
                MettaValue::Long(42),
            ]),
            MettaValue::Atom("$x".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("Hint"), "Expected 'Hint' in: {}", msg);
                assert!(
                    msg.contains("foo") && msg.contains("bar"),
                    "Expected head mismatch hint in: {}",
                    msg
                );
            }
            _ => panic!("Expected error with pattern mismatch hint"),
        }
    }

    #[test]
    fn test_pattern_mismatch_literal_hint() {
        let env = Environment::new();

        // (let (pair 42 $x) (pair 99 hello) $x) - literal 42 doesn't match 99
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("pair".to_string()),
                MettaValue::Long(42),
                MettaValue::Atom("$x".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("pair".to_string()),
                MettaValue::Long(99),
                MettaValue::Atom("hello".to_string()),
            ]),
            MettaValue::Atom("$x".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("Hint"), "Expected 'Hint' in: {}", msg);
                // Should mention the position and values that don't match
                assert!(
                    msg.contains("position") || msg.contains("doesn't match"),
                    "Expected position mismatch hint in: {}",
                    msg
                );
            }
            _ => panic!("Expected error with pattern mismatch hint"),
        }
    }

    #[test]
    fn test_let_with_mixed_pattern_elements() {
        let env = Environment::new();

        // Pattern with mix of literals and variables
        // (let (mixed 42 $x "literal" $y) (mixed 42 100 "literal" 200) (+ $x $y))
        let mixed_pattern = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("mixed".to_string()),
                MettaValue::Long(42),
                MettaValue::Atom("$x".to_string()),
                MettaValue::String("literal".to_string()),
                MettaValue::Atom("$y".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("mixed".to_string()),
                MettaValue::Long(42),
                MettaValue::Long(100),
                MettaValue::String("literal".to_string()),
                MettaValue::Long(200),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Atom("$y".to_string()),
            ]),
        ]);

        let (results, _) = eval(mixed_pattern, env.clone());
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(300)); // 100 + 200

        // Test failure case where literal doesn't match
        // (let (mixed 42 $x "literal" $y) (mixed 43 100 "literal" 200) (+ $x $y))
        let mixed_fail = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("mixed".to_string()),
                MettaValue::Long(42),
                MettaValue::Atom("$x".to_string()),
                MettaValue::String("literal".to_string()),
                MettaValue::Atom("$y".to_string()),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("mixed".to_string()),
                MettaValue::Long(43), // Different literal - should fail
                MettaValue::Long(100),
                MettaValue::String("literal".to_string()),
                MettaValue::Long(200),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Atom("$x".to_string()),
                MettaValue::Atom("$y".to_string()),
            ]),
        ]);

        let (results, _) = eval(mixed_fail, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("does not match"));
            }
            _ => panic!("Expected pattern match error"),
        }
    }

    #[test]
    fn test_let_with_complex_body_expressions() {
        let env = Environment::new();

        // Test let with complex body containing multiple operations
        // (let $base 5
        //   (if (> $base 0)
        //     (let $squared (* $base $base)
        //       (+ $squared $base))
        //     0))
        let complex_body = MettaValue::SExpr(vec![
            MettaValue::Atom("let".to_string()),
            MettaValue::Atom("$base".to_string()),
            MettaValue::Long(5),
            MettaValue::SExpr(vec![
                MettaValue::Atom("if".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom(">".to_string()),
                    MettaValue::Atom("$base".to_string()),
                    MettaValue::Long(0),
                ]),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("let".to_string()),
                    MettaValue::Atom("$squared".to_string()),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("*".to_string()),
                        MettaValue::Atom("$base".to_string()),
                        MettaValue::Atom("$base".to_string()),
                    ]),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("+".to_string()),
                        MettaValue::Atom("$squared".to_string()),
                        MettaValue::Atom("$base".to_string()),
                    ]),
                ]),
                MettaValue::Long(0),
            ]),
        ]);

        let (results, _) = eval(complex_body, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(30)); // (5 * 5) + 5 = 30
    }
}
