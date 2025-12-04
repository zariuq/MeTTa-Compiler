use crate::backend::environment::Environment;
use crate::backend::models::{EvalResult, MettaValue};
use std::sync::Arc;

use super::{apply_bindings, eval, pattern_match};

/// Evaluate if control flow: (if condition then-branch else-branch)
/// Only evaluates the chosen branch (lazy evaluation)
pub(super) fn eval_if(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.len() < 3 {
        let got = args.len();
        let err = MettaValue::Error(
            format!(
                "if requires exactly 3 arguments, got {}. Usage: (if condition then-branch else-branch)",
                got
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    let condition = &args[0];
    let then_branch = &args[1];
    let else_branch = &args[2];

    // Evaluate the condition
    let (cond_results, env_after_cond) = eval(condition.clone(), env);

    // Check for error in condition
    if let Some(first) = cond_results.first() {
        if matches!(first, MettaValue::Error(_, _)) {
            return (vec![first.clone()], env_after_cond);
        }

        // Check if condition is true
        let is_true = match first {
            MettaValue::Bool(true) => true,
            MettaValue::Bool(false) => false,
            // Non-boolean values: treat as true if not Nil
            MettaValue::Nil => false,
            _ => true,
        };

        // Evaluate only the chosen branch
        if is_true {
            eval(then_branch.clone(), env_after_cond)
        } else {
            eval(else_branch.clone(), env_after_cond)
        }
    } else {
        // No result from condition - treat as false
        eval(else_branch.clone(), env_after_cond)
    }
}

/// Select first k results from an expression evaluation
/// Usage: (select k expr)
/// Similar to Prolog's once/1 (when k=1), but generalized to select first k results
pub(super) fn eval_select(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.len() < 2 {
        let got = args.len();
        let err = MettaValue::Error(
            format!(
                "select requires exactly 2 arguments, got {}. Usage: (select k expr)",
                got
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    let k_expr = &args[0];
    let expr = &args[1];

    // Evaluate k to get the number
    let (k_results, k_env) = eval(k_expr.clone(), env);

    // Check for error in k
    if let Some(first) = k_results.first() {
        if matches!(first, MettaValue::Error(_, _)) {
            return (vec![first.clone()], k_env);
        }

        // Extract k as usize
        let k_val = match first {
            MettaValue::Long(n) => {
                if *n < 0 {
                    let err = MettaValue::Error(
                        format!("select: k must be non-negative, got {}", n),
                        Arc::new(first.clone()),
                    );
                    return (vec![err], k_env);
                }
                *n as usize
            }
            MettaValue::Float(f) => {
                if *f < 0.0 {
                    let err = MettaValue::Error(
                        format!("select: k must be non-negative, got {}", f),
                        Arc::new(first.clone()),
                    );
                    return (vec![err], k_env);
                }
                *f as usize
            }
            _ => {
                let err = MettaValue::Error(
                    format!("select: k must be a number, got {:?}", first),
                    Arc::new(first.clone()),
                );
                return (vec![err], k_env);
            }
        };

        // Evaluate the expression and truncate to first k results
        let (mut results, final_env) = eval(expr.clone(), k_env);
        results.truncate(k_val);
        (results, final_env)
    } else {
        // No result from k - return error
        let err = MettaValue::Error(
            "select: k expression produced no results".to_string(),
            Arc::new(k_expr.clone()),
        );
        (vec![err], k_env)
    }
}

/// Superpose: takes a single S-expression and produces multiple results (one per element)
/// Usage: (superpose (A B C D)) → {A, B, C, D}
/// This is the inverse of collapse - it "explodes" a list into multiple branches
pub(super) fn eval_superpose(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.len() != 1 {
        let got = args.len();
        let err = MettaValue::Error(
            format!(
                "superpose requires exactly 1 argument, got {}. Usage: (superpose (A B C D))",
                got
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    let expr = &args[0];

    // Evaluate the argument to get the list
    let (list_results, list_env) = eval(expr.clone(), env);

    // Check for error in evaluation
    if let Some(first) = list_results.first() {
        if matches!(first, MettaValue::Error(_, _)) {
            return (vec![first.clone()], list_env);
        }
    }

    // Extract first result (should be an S-expression)
    match list_results.first() {
        Some(MettaValue::SExpr(elements)) => {
            // Return each element as a separate result (scatter operation)
            (elements.clone(), list_env)
        }
        Some(MettaValue::Nil) => {
            // Nil (empty S-expression) returns 0 results
            (vec![], list_env)
        }
        Some(other) => {
            let err = MettaValue::Error(
                format!("superpose expects S-expression, got {:?}", other),
                Arc::new(other.clone()),
            );
            (vec![err], list_env)
        }
        None => {
            // Empty result - return empty vec
            (vec![], list_env)
        }
    }
}

/// Collapse: collects all alternatives from an expression into a single S-expression
/// Usage: (collapse (color)) where (color) → {red, green, blue}
/// Returns: ONE result containing (red green blue)
/// This is the inverse of superpose - it "gathers" multiple branches into one list
pub(super) fn eval_collapse(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    let args = &items[1..];

    if args.len() != 1 {
        let got = args.len();
        let err = MettaValue::Error(
            format!(
                "collapse requires exactly 1 argument, got {}. Usage: (collapse expr)",
                got
            ),
            Arc::new(MettaValue::SExpr(args.to_vec())),
        );
        return (vec![err], env);
    }

    let expr = &args[0];

    // Evaluate the expression to get ALL results (gather operation)
    let (results, final_env) = eval(expr.clone(), env);

    // Wrap ALL results into ONE S-expression
    let collapsed = MettaValue::SExpr(results);
    (vec![collapsed], final_env) // Return exactly ONE result!
}

/// Subsequently tests multiple pattern-matching conditions (second argument) for the
/// given value (first argument)
pub(super) fn eval_case(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    require_args_with_usage!(
        "case",
        items,
        2,
        env,
        "(case expr ((pattern1 result1) ...))"
    );

    let atom = items[1].clone();
    let cases = items[2].clone();

    let (atom_results, atom_env) = eval(atom, env);
    let mut final_results = Vec::new();

    for atom_result in atom_results {
        let is_empty = match &atom_result {
            MettaValue::Nil => true,
            MettaValue::SExpr(items) if items.is_empty() => true,
            _ => false,
        };

        if is_empty {
            let switch_result = eval_switch_minimal(
                MettaValue::Atom("Empty".to_string()),
                cases.clone(),
                atom_env.clone(),
            );
            final_results.extend(switch_result.0);
        } else {
            let switch_result = eval_switch_minimal(atom_result, cases.clone(), atom_env.clone());
            final_results.extend(switch_result.0);
        }
    }

    (final_results, atom_env)
}

/// Difference between `switch` and `case` is a way how they interpret `Empty` result.
/// case interprets first argument inside itself and then manually checks whether result is empty.
pub(super) fn eval_switch(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    require_args_with_usage!(
        "switch",
        items,
        2,
        env,
        "(switch expr ((pattern1 result1) ...))"
    );
    let atom = items[1].clone();
    let cases = items[2].clone();
    eval_switch_minimal(atom, cases, env)
}

pub(super) fn eval_switch_minimal_handler(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    require_args_with_usage!(
        "switch-minimal",
        items,
        2,
        env,
        "(switch-minimal expr cases)"
    );
    let atom = items[1].clone();
    let cases = items[2].clone();
    eval_switch_minimal(atom, cases, env)
}

/// This function is being called inside switch function to test one of the cases and it
/// calls switch once again if current condition is not met
pub(super) fn eval_switch_internal_handler(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    require_args_with_usage!(
        "switch-internal",
        items,
        2,
        env,
        "(switch-internal expr cases-data)"
    );
    let atom = items[1].clone();
    let cases = items[2].clone();
    eval_switch_internal(atom, cases, env)
}

/// Helper function to implement switch-minimal logic
/// Handles the main switch logic by deconstructing cases and calling switch-internal
fn eval_switch_minimal(atom: MettaValue, cases: MettaValue, env: Environment) -> EvalResult {
    if let MettaValue::SExpr(cases_items) = cases {
        if cases_items.is_empty() {
            return (vec![MettaValue::Atom("NotReducible".to_string())], env);
        }

        let first_case = cases_items[0].clone();
        let remaining_cases = if cases_items.len() > 1 {
            MettaValue::SExpr(cases_items[1..].to_vec())
        } else {
            MettaValue::SExpr(vec![])
        };

        let cases_list = MettaValue::SExpr(vec![first_case, remaining_cases]);
        return eval_switch_internal(atom, cases_list, env);
    }

    let err = MettaValue::Error(
        format!(
            "switch-minimal expects expression as second argument, got: {}",
            super::friendly_value_repr(&cases)
        ),
        Arc::new(cases),
    );
    (vec![err], env)
}

/// Helper function to implement switch-internal logic
/// Tests one case and recursively tries remaining cases if no match
fn eval_switch_internal(atom: MettaValue, cases_data: MettaValue, env: Environment) -> EvalResult {
    if let MettaValue::SExpr(cases_items) = cases_data {
        if cases_items.len() != 2 {
            let err = MettaValue::Error(
                format!(
                    "switch-internal expects exactly 2 arguments, got {}. \
                     Usage: (switch-internal expr (first-case remaining-cases))",
                    cases_items.len()
                ),
                Arc::new(MettaValue::SExpr(cases_items)),
            );
            return (vec![err], env);
        }

        let first_case = cases_items[0].clone();
        let remaining_cases = cases_items[1].clone();

        if let MettaValue::SExpr(case_items) = first_case {
            if case_items.len() != 2 {
                let err = MettaValue::Error(
                    format!(
                        "switch case should be a pattern-template pair with exactly 2 elements, got {}. \
Usage: (switch expr (pattern1 result1) (pattern2 result2) ...)",
                        case_items.len()
                    ),
                    Arc::new(MettaValue::SExpr(case_items)),
                );
                return (vec![err], env);
            }

            let pattern = case_items[0].clone();
            let template = case_items[1].clone();

            if let Some(bindings) = pattern_match(&pattern, &atom) {
                let instantiated_template = apply_bindings(&template, &bindings);
                return eval(instantiated_template, env);
            } else {
                return eval_switch_minimal(atom, remaining_cases, env);
            }
        } else {
            let err = MettaValue::Error(
                format!(
                    "switch case should be an expression (pattern-template pair), got: {}",
                    super::friendly_value_repr(&first_case)
                ),
                Arc::new(first_case),
            );
            return (vec![err], env);
        }
    }

    let err = MettaValue::Error(
        format!(
            "switch-internal expects expression argument, got: {}",
            super::friendly_value_repr(&cases_data)
        ),
        Arc::new(cases_data),
    );
    (vec![err], env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::models::Rule;

    #[test]
    fn test_if_true_branch() {
        let env = Environment::new();

        // (if true (+ 1 2) (+ 3 4))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Bool(true),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(3),
                MettaValue::Long(4),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(3)); // 1 + 2
    }

    #[test]
    fn test_if_false_branch() {
        let env = Environment::new();

        // (if false (+ 1 2) (+ 3 4))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Bool(false),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(3),
                MettaValue::Long(4),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(7)); // 3 + 4
    }

    #[test]
    fn test_if_with_comparison() {
        let env = Environment::new();

        // (if (< 1 2) "yes" "no")
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("<".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
            MettaValue::String("yes".to_string()),
            MettaValue::String("no".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("yes".to_string()));
    }

    #[test]
    fn test_if_only_evaluates_chosen_branch() {
        let env = Environment::new();

        // (if true 1 (error "should not evaluate"))
        // The error in the else branch should not be evaluated
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Bool(true),
            MettaValue::Long(1),
            MettaValue::SExpr(vec![
                MettaValue::Atom("error".to_string()),
                MettaValue::String("should not evaluate".to_string()),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(1)); // No error!
    }

    #[test]
    fn test_switch_basic() {
        let env = Environment::new();

        // (switch 42 ((42 "found") (43 "not found")))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![
                // First case: (42 "found")
                MettaValue::SExpr(vec![
                    MettaValue::Long(42),
                    MettaValue::String("found".to_string()),
                ]),
                // Second case: (43 "not found")
                MettaValue::SExpr(vec![
                    MettaValue::Long(43),
                    MettaValue::String("not found".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("found".to_string()));
    }

    #[test]
    fn test_switch_with_variables() {
        let env = Environment::new();

        // (switch 42 (($x (+ $x 10))))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![
                // Case: ($x (+ $x 10))
                MettaValue::SExpr(vec![
                    MettaValue::Atom("$x".to_string()),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("+".to_string()),
                        MettaValue::Atom("$x".to_string()),
                        MettaValue::Long(10),
                    ]),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(52)); // 42 + 10
    }

    #[test]
    fn test_switch_no_match() {
        let env = Environment::new();

        // (switch 50 ((42 "found") (43 "not found")))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(50),
            MettaValue::SExpr(vec![
                // First case: (42 "found")
                MettaValue::SExpr(vec![
                    MettaValue::Long(42),
                    MettaValue::String("found".to_string()),
                ]),
                // Second case: (43 "not found")
                MettaValue::SExpr(vec![
                    MettaValue::Long(43),
                    MettaValue::String("not found".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Atom("NotReducible".to_string()));
    }

    #[test]
    fn test_switch_with_sexpr_pattern() {
        let env = Environment::new();

        // (switch (foo 42) (((foo $x) (+ $x 1)) ((bar $y) (+ $y 2))))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("foo".to_string()),
                MettaValue::Long(42),
            ]),
            MettaValue::SExpr(vec![
                // First case: ((foo $x) (+ $x 1))
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("foo".to_string()),
                        MettaValue::Atom("$x".to_string()),
                    ]),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("+".to_string()),
                        MettaValue::Atom("$x".to_string()),
                        MettaValue::Long(1),
                    ]),
                ]),
                // Second case: ((bar $y) (+ $y 2))
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("bar".to_string()),
                        MettaValue::Atom("$y".to_string()),
                    ]),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("+".to_string()),
                        MettaValue::Atom("$y".to_string()),
                        MettaValue::Long(2),
                    ]),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(43)); // 42 + 1
    }

    #[test]
    fn test_case_basic() {
        let env = Environment::new();

        // (case 42 ((42 "found") (43 "not found")))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("case".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![
                // First case: (42 "found")
                MettaValue::SExpr(vec![
                    MettaValue::Long(42),
                    MettaValue::String("found".to_string()),
                ]),
                // Second case: (43 "not found")
                MettaValue::SExpr(vec![
                    MettaValue::Long(43),
                    MettaValue::String("not found".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("found".to_string()));
    }

    #[test]
    fn test_case_with_evaluation() {
        let env = Environment::new();

        // (case (+ 1 2) ((3 "three") (4 "four")))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("case".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
            MettaValue::SExpr(vec![
                // First case: (3 "three")
                MettaValue::SExpr(vec![
                    MettaValue::Long(3),
                    MettaValue::String("three".to_string()),
                ]),
                // Second case: (4 "four")
                MettaValue::SExpr(vec![
                    MettaValue::Long(4),
                    MettaValue::String("four".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("three".to_string()));
    }

    #[test]
    fn test_case_with_empty_result() {
        let mut env = Environment::new();

        // First define a rule that returns empty: (= (empty-result) ())
        let empty_rule = Rule {
            lhs: MettaValue::SExpr(vec![MettaValue::Atom("empty-result".to_string())]),
            rhs: MettaValue::SExpr(vec![]),
        };
        env.add_rule(empty_rule);

        // (case (empty-result) ((Empty "was empty") (42 "was forty-two")))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("case".to_string()),
            MettaValue::SExpr(vec![MettaValue::Atom("empty-result".to_string())]),
            MettaValue::SExpr(vec![
                // First case: (Empty "was empty")
                MettaValue::SExpr(vec![
                    MettaValue::Atom("Empty".to_string()),
                    MettaValue::String("was empty".to_string()),
                ]),
                // Second case: (42 "was forty-two")
                MettaValue::SExpr(vec![
                    MettaValue::Long(42),
                    MettaValue::String("was forty-two".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("was empty".to_string()));
    }

    #[test]
    fn test_switch_first_match_wins() {
        let env = Environment::new();

        // (switch 42 (($x "first") (42 "second")))
        // Should match the first case ($x matches anything)
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![
                // First case: ($x "first") - variable matches anything
                MettaValue::SExpr(vec![
                    MettaValue::Atom("$x".to_string()),
                    MettaValue::String("first".to_string()),
                ]),
                // Second case: (42 "second") - literal match
                MettaValue::SExpr(vec![
                    MettaValue::Long(42),
                    MettaValue::String("second".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("first".to_string()));
    }

    #[test]
    fn test_switch_empty_cases() {
        let env = Environment::new();

        // (switch 42 ())
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![]), // Empty cases
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Atom("NotReducible".to_string()));
    }

    #[test]
    fn test_switch_with_wildcard() {
        let env = Environment::new();

        // (switch 42 ((100 "hundred") (_ "anything else")))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![
                // First case: (100 "hundred")
                MettaValue::SExpr(vec![
                    MettaValue::Long(100),
                    MettaValue::String("hundred".to_string()),
                ]),
                // Second case: (_ "anything else") - wildcard
                MettaValue::SExpr(vec![
                    MettaValue::Atom("_".to_string()),
                    MettaValue::String("anything else".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("anything else".to_string()));
    }

    #[test]
    fn test_switch_missing_arguments() {
        let env = Environment::new();

        // (switch) - missing both arguments
        let value = MettaValue::SExpr(vec![MettaValue::Atom("switch".to_string())]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("switch"));
                assert!(msg.contains("requires exactly 2 arguments"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_case_missing_arguments() {
        let env = Environment::new();

        // (case 42) - missing cases argument
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("case".to_string()),
            MettaValue::Long(42),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("case"));
                assert!(msg.contains("requires exactly 2 arguments"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_switch_malformed_case() {
        let env = Environment::new();

        // (switch 42 ((42))) - case missing template
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::Long(42),
            MettaValue::SExpr(vec![
                // Malformed case: (42) - missing template
                MettaValue::SExpr(vec![MettaValue::Long(42)]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("pattern-template pair"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_switch_with_complex_patterns() {
        let env = Environment::new();

        // (switch (add 10 20) (((add $x $y) (+ $x $y $x)) ((mul $a $b) (* $a $b))))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("add".to_string()),
                MettaValue::Long(10),
                MettaValue::Long(20),
            ]),
            MettaValue::SExpr(vec![
                // First case: ((add $x $y) (+ $x $y $x))
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("add".to_string()),
                        MettaValue::Atom("$x".to_string()),
                        MettaValue::Atom("$y".to_string()),
                    ]),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("+".to_string()),
                        MettaValue::Atom("$x".to_string()),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("+".to_string()),
                            MettaValue::Atom("$y".to_string()),
                            MettaValue::Atom("$x".to_string()),
                        ]),
                    ]),
                ]),
                // Second case: ((mul $a $b) (* $a $b))
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("mul".to_string()),
                        MettaValue::Atom("$a".to_string()),
                        MettaValue::Atom("$b".to_string()),
                    ]),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("*".to_string()),
                        MettaValue::Atom("$a".to_string()),
                        MettaValue::Atom("$b".to_string()),
                    ]),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(40)); // 10 + (20 + 10) = 40
    }

    #[test]
    fn test_switch_vs_case_empty_handling() {
        let mut env = Environment::new();

        // Define a rule that can return Empty: (= (maybe-empty $x) (if (== $x 0) () $x))
        let maybe_empty_rule = Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("maybe-empty".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::SExpr(vec![
                MettaValue::Atom("if".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("==".to_string()),
                    MettaValue::Atom("$x".to_string()),
                    MettaValue::Long(0),
                ]),
                MettaValue::SExpr(vec![]), // Empty s-expression
                MettaValue::Atom("$x".to_string()),
            ]),
        };
        env.add_rule(maybe_empty_rule);

        // Test switch: does NOT evaluate first argument
        // (switch (maybe-empty 0) (((maybe-empty $y) "matched") (Empty "was empty")))
        let switch_test = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("maybe-empty".to_string()),
                MettaValue::Long(0),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("maybe-empty".to_string()),
                        MettaValue::Atom("$y".to_string()),
                    ]),
                    MettaValue::String("matched".to_string()),
                ]),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("Empty".to_string()),
                    MettaValue::String("was empty".to_string()),
                ]),
            ]),
        ]);

        let (switch_results, env2) = eval(switch_test, env.clone());
        assert_eq!(switch_results.len(), 1);
        assert_eq!(switch_results[0], MettaValue::String("matched".to_string()));

        // Test case: DOES evaluate first argument
        // (case (maybe-empty 0) (((maybe-empty $y) "matched") (Empty "was empty")))
        let case_test = MettaValue::SExpr(vec![
            MettaValue::Atom("case".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("maybe-empty".to_string()),
                MettaValue::Long(0),
            ]),
            MettaValue::SExpr(vec![
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("maybe-empty".to_string()),
                        MettaValue::Atom("$y".to_string()),
                    ]),
                    MettaValue::String("matched".to_string()),
                ]),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("Empty".to_string()),
                    MettaValue::String("was empty".to_string()),
                ]),
            ]),
        ]);

        let (case_results, _) = eval(case_test, env2);
        assert_eq!(case_results.len(), 1);
        assert_eq!(case_results[0], MettaValue::String("was empty".to_string()));
    }

    /// Test that lowercase "true"/"false" atoms are NOT treated as boolean values.
    /// This is critical: MeTTa uses uppercase True/False for booleans.
    /// Lowercase true/false are just atoms and should be truthy (like any other atom).
    #[test]
    fn test_if_lowercase_false_is_truthy_atom() {
        let env = Environment::new();

        // (if false 1 2) should return 1 because "false" is an atom (truthy), not Bool(false)
        // This mirrors: !(if false yes no) → [yes] (atom "false" is truthy!)
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Atom("false".to_string()), // lowercase = atom, NOT boolean
            MettaValue::Long(1),
            MettaValue::Long(2),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(1)); // "false" atom is truthy!
    }

    #[test]
    fn test_if_lowercase_true_is_truthy_atom() {
        let env = Environment::new();

        // (if true 1 2) should also return 1 because "true" is an atom (truthy)
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Atom("true".to_string()), // lowercase = atom, truthy
            MettaValue::Long(1),
            MettaValue::Long(2),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(1)); // atom is truthy
    }

    #[test]
    fn test_if_uppercase_bool_true_works() {
        let env = Environment::new();

        // (if True 1 2) with MettaValue::Bool(true) should return 1
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Bool(true), // Proper boolean True
            MettaValue::Long(1),
            MettaValue::Long(2),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(1));
    }

    #[test]
    fn test_if_uppercase_bool_false_works() {
        let env = Environment::new();

        // (if False 1 2) with MettaValue::Bool(false) should return 2
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Bool(false), // Proper boolean False
            MettaValue::Long(1),
            MettaValue::Long(2),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(2)); // Takes else branch
    }

    #[test]
    fn test_if_with_equality_returns_proper_bool() {
        let env = Environment::new();

        // (if (== 1 1) "yes" "no") should return "yes"
        // This tests that == returns MettaValue::Bool(true), not an atom
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("==".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(1),
            ]),
            MettaValue::String("yes".to_string()),
            MettaValue::String("no".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("yes".to_string()));
    }

    #[test]
    fn test_if_with_inequality_returns_proper_bool() {
        let env = Environment::new();

        // (if (== 1 2) "yes" "no") should return "no"
        // This tests that == returns MettaValue::Bool(false), not an atom
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("==".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
            MettaValue::String("yes".to_string()),
            MettaValue::String("no".to_string()),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::String("no".to_string()));
    }

    #[test]
    fn test_if_nil_is_falsy() {
        let env = Environment::new();

        // (if () 1 2) should return 2 because () is falsy
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("if".to_string()),
            MettaValue::Nil,
            MettaValue::Long(1),
            MettaValue::Long(2),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(2)); // Nil is falsy
    }

    #[test]
    fn test_switch_case_with_nested_pattern_matching_and_variable_scoping() {
        let env = Environment::new();

        // Test complex nested pattern matching with variable consistency
        // Create a test structure directly in the test
        let test_structure = MettaValue::SExpr(vec![
            MettaValue::Atom("complex".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("nested".to_string()),
                MettaValue::Long(5),
                MettaValue::Long(5), // Same value as previous - should match same variable
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("nested".to_string()),
                MettaValue::Long(5), // Same again
                MettaValue::Long(7), // Different value
            ]),
        ]);

        // Test switch with pattern that requires variable consistency
        let complex_switch = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            test_structure,
            MettaValue::SExpr(vec![
                // Case 1: Pattern with variable consistency requirement
                // ((complex (nested $x $x) (nested $x $y)) (+ $x $y))
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("complex".to_string()),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("nested".to_string()),
                            MettaValue::Atom("$x".to_string()),
                            MettaValue::Atom("$x".to_string()), // Must match same value
                        ]),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("nested".to_string()),
                            MettaValue::Atom("$x".to_string()), // Must match same $x as above
                            MettaValue::Atom("$y".to_string()), // Can be different
                        ]),
                    ]),
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("+".to_string()),
                        MettaValue::Atom("$x".to_string()),
                        MettaValue::Atom("$y".to_string()),
                    ]),
                ]),
                // Case 2: Different pattern that shouldn't match
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("complex".to_string()),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("nested".to_string()),
                            MettaValue::Atom("$a".to_string()),
                            MettaValue::Atom("$b".to_string()), // Different variables
                        ]),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("nested".to_string()),
                            MettaValue::Atom("$c".to_string()),
                            MettaValue::Atom("$d".to_string()),
                        ]),
                    ]),
                    MettaValue::String("different pattern".to_string()),
                ]),
                // Case 3: Fallback
                MettaValue::SExpr(vec![
                    MettaValue::Atom("_".to_string()),
                    MettaValue::String("no match".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(complex_switch, env.clone());
        assert_eq!(results.len(), 1);
        // Structure: (complex (nested 5 5) (nested 5 7))
        // Pattern: (complex (nested $x $x) (nested $x $y))
        // Bindings: $x=5, $y=7, result is 5+7=12
        assert_eq!(results[0], MettaValue::Long(12));

        // Test 2: Variable consistency failure case
        let test_structure_inconsistent = MettaValue::SExpr(vec![
            MettaValue::Atom("complex".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("nested".to_string()),
                MettaValue::Long(5),
                MettaValue::Long(3), // Different values
            ]),
            MettaValue::SExpr(vec![
                MettaValue::Atom("nested".to_string()),
                MettaValue::Long(5),
                MettaValue::Long(7),
            ]),
        ]);

        let complex_switch_fail = MettaValue::SExpr(vec![
            MettaValue::Atom("switch".to_string()),
            test_structure_inconsistent,
            MettaValue::SExpr(vec![
                // Same pattern as before - should NOT match because first nested has (5,3) not (5,5)
                MettaValue::SExpr(vec![
                    MettaValue::SExpr(vec![
                        MettaValue::Atom("complex".to_string()),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("nested".to_string()),
                            MettaValue::Atom("$x".to_string()),
                            MettaValue::Atom("$x".to_string()), // Would require 5=3, which is false
                        ]),
                        MettaValue::SExpr(vec![
                            MettaValue::Atom("nested".to_string()),
                            MettaValue::Atom("$x".to_string()),
                            MettaValue::Atom("$y".to_string()),
                        ]),
                    ]),
                    MettaValue::String("should not match".to_string()),
                ]),
                // Fallback should match
                MettaValue::SExpr(vec![
                    MettaValue::Atom("_".to_string()),
                    MettaValue::String("fallback matched".to_string()),
                ]),
            ]),
        ]);

        let (results2, _) = eval(complex_switch_fail, env);
        assert_eq!(results2.len(), 1);
        assert_eq!(
            results2[0],
            MettaValue::String("fallback matched".to_string())
        );
    }

    // ============================================================================
    // select tests
    // ============================================================================

    #[test]
    fn test_select_basic_first() {
        let mut env = Environment::new();

        // Define a function that returns multiple results
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(1),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(2),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(3),
        });

        // (select 1 (multi 42)) should return only [1]
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(1),
            MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Long(42),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(1));
    }

    #[test]
    fn test_select_first_two() {
        let mut env = Environment::new();

        // Define a function that returns multiple results
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(1),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(2),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(3),
        });

        // (select 2 (multi 42)) should return [1, 2]
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(2),
            MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Long(42),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], MettaValue::Long(1));
        assert_eq!(results[1], MettaValue::Long(2));
    }

    #[test]
    fn test_select_zero_results() {
        let env = Environment::new();

        // (select 0 (+ 1 2)) should return []
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(0),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_select_more_than_available() {
        let mut env = Environment::new();

        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(1),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Long(2),
        });

        // (select 10 (multi 42)) should return [1, 2] (all available)
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(10),
            MettaValue::SExpr(vec![
                MettaValue::Atom("multi".to_string()),
                MettaValue::Long(42),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], MettaValue::Long(1));
        assert_eq!(results[1], MettaValue::Long(2));
    }

    #[test]
    fn test_select_with_float_k() {
        let env = Environment::new();

        // (select 2.7 (+ 1 2)) should truncate to 2
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Float(2.7),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Long(3));
    }

    #[test]
    fn test_select_negative_k_error() {
        let env = Environment::new();

        // (select -1 (+ 1 2)) should return error
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(-1),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("non-negative"));
            }
            _ => panic!("Expected error for negative k"),
        }
    }

    #[test]
    fn test_select_non_numeric_k_error() {
        let env = Environment::new();

        // (select "hello" (+ 1 2)) should return error
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::String("hello".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(1),
                MettaValue::Long(2),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("must be a number"));
            }
            _ => panic!("Expected error for non-numeric k"),
        }
    }

    #[test]
    fn test_select_missing_arguments() {
        let env = Environment::new();

        // (select 1) - missing expression
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(1),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("requires exactly 2 arguments"));
            }
            _ => panic!("Expected error for missing arguments"),
        }
    }

    #[test]
    fn test_select_no_expression() {
        let env = Environment::new();

        // (select) - no arguments at all
        let value = MettaValue::SExpr(vec![MettaValue::Atom("select".to_string())]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("requires exactly 2 arguments"));
                assert!(msg.contains("got 0"));
            }
            _ => panic!("Expected error for no arguments"),
        }
    }

    #[test]
    fn test_select_with_expression_returning_one_result() {
        let env = Environment::new();

        // Test with an undefined function that returns the expression unevaluated
        let undefined_expr = MettaValue::SExpr(vec![
            MettaValue::Atom("undefined-function".to_string()),
            MettaValue::Long(42),
        ]);
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(5),
            undefined_expr.clone(),
        ]);

        let (results, _) = eval(value, env);
        // Undefined function returns the expression itself, select returns 1 result
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], undefined_expr);
    }

    #[test]
    fn test_select_prolog_once_style() {
        let mut env = Environment::new();

        // Like Prolog's once/1 - select 1 for deterministic choice
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("choice".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Atom("first".to_string()),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("choice".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Atom("second".to_string()),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("choice".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Atom("third".to_string()),
        });

        // (select 1 (choice foo)) - Prolog once/1 style
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(1),
            MettaValue::SExpr(vec![
                MettaValue::Atom("choice".to_string()),
                MettaValue::Atom("foo".to_string()),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Atom("first".to_string()));
    }

    // ===== Tests for superpose =====

    #[test]
    fn test_superpose_basic() {
        let env = Environment::new();

        // (superpose (A B C D))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("superpose".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("A".to_string()),
                MettaValue::Atom("B".to_string()),
                MettaValue::Atom("C".to_string()),
                MettaValue::Atom("D".to_string()),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 4);
        assert_eq!(results[0], MettaValue::Atom("A".to_string()));
        assert_eq!(results[1], MettaValue::Atom("B".to_string()));
        assert_eq!(results[2], MettaValue::Atom("C".to_string()));
        assert_eq!(results[3], MettaValue::Atom("D".to_string()));
    }

    #[test]
    fn test_superpose_with_nested_sexprs() {
        let env = Environment::new();

        // (superpose ((a 1) (b 2) (c 3)))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("superpose".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::SExpr(vec![
                    MettaValue::Atom("a".to_string()),
                    MettaValue::Long(1),
                ]),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("b".to_string()),
                    MettaValue::Long(2),
                ]),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("c".to_string()),
                    MettaValue::Long(3),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 3);
        assert!(matches!(results[0], MettaValue::SExpr(_)));
    }

    #[test]
    fn test_superpose_empty_list() {
        let env = Environment::new();

        // (superpose ())
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("superpose".to_string()),
            MettaValue::SExpr(vec![]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_superpose_single_element() {
        let env = Environment::new();

        // (superpose (X))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("superpose".to_string()),
            MettaValue::SExpr(vec![MettaValue::Atom("X".to_string())]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], MettaValue::Atom("X".to_string()));
    }

    #[test]
    fn test_superpose_wrong_args() {
        let env = Environment::new();

        // (superpose) - no arguments
        let value = MettaValue::SExpr(vec![MettaValue::Atom("superpose".to_string())]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("requires exactly 1 argument"));
            }
            _ => panic!("Expected error for missing arguments"),
        }
    }

    #[test]
    fn test_superpose_non_sexpr_input() {
        let env = Environment::new();

        // (superpose 42) - not an S-expression
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("superpose".to_string()),
            MettaValue::Long(42),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("expects S-expression"));
            }
            _ => panic!("Expected error for non-SExpr input"),
        }
    }

    // ===== Tests for collapse =====

    #[test]
    fn test_collapse_basic() {
        let mut env = Environment::new();

        // Define multiple matching rules
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("color".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Atom("red".to_string()),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("color".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Atom("green".to_string()),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![
                MettaValue::Atom("color".to_string()),
                MettaValue::Atom("$x".to_string()),
            ]),
            rhs: MettaValue::Atom("blue".to_string()),
        });

        // (collapse (color apple))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("collapse".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("color".to_string()),
                MettaValue::Atom("apple".to_string()),
            ]),
        ]);

        let (results, _) = eval(value, env);
        // Should return exactly ONE result, which is an S-expression containing all three colors
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::SExpr(elements) => {
                assert_eq!(elements.len(), 3);
                assert_eq!(elements[0], MettaValue::Atom("red".to_string()));
                assert_eq!(elements[1], MettaValue::Atom("green".to_string()));
                assert_eq!(elements[2], MettaValue::Atom("blue".to_string()));
            }
            _ => panic!("Expected S-expression result from collapse"),
        }
    }

    #[test]
    fn test_collapse_single_result() {
        let env = Environment::new();

        // (collapse (+ 2 3)) - single result
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("collapse".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("+".to_string()),
                MettaValue::Long(2),
                MettaValue::Long(3),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::SExpr(elements) => {
                assert_eq!(elements.len(), 1);
                assert_eq!(elements[0], MettaValue::Long(5));
            }
            _ => panic!("Expected S-expression result from collapse"),
        }
    }

    #[test]
    fn test_collapse_empty_result() {
        let env = Environment::new();

        // (collapse (undefined-function)) - no results
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("collapse".to_string()),
            MettaValue::SExpr(vec![MettaValue::Atom("undefined-function".to_string())]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::SExpr(elements) => {
                // Undefined function returns the unevaluated expression
                assert_eq!(elements.len(), 1);
            }
            _ => panic!("Expected S-expression result from collapse"),
        }
    }

    #[test]
    fn test_collapse_superpose_roundtrip() {
        let env = Environment::new();

        // (collapse (superpose (X Y Z))) should return (X Y Z)
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("collapse".to_string()),
            MettaValue::SExpr(vec![
                MettaValue::Atom("superpose".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("X".to_string()),
                    MettaValue::Atom("Y".to_string()),
                    MettaValue::Atom("Z".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::SExpr(elements) => {
                assert_eq!(elements.len(), 3);
                assert_eq!(elements[0], MettaValue::Atom("X".to_string()));
                assert_eq!(elements[1], MettaValue::Atom("Y".to_string()));
                assert_eq!(elements[2], MettaValue::Atom("Z".to_string()));
            }
            _ => panic!("Expected S-expression result from collapse"),
        }
    }

    #[test]
    fn test_collapse_wrong_args() {
        let env = Environment::new();

        // (collapse) - no arguments
        let value = MettaValue::SExpr(vec![MettaValue::Atom("collapse".to_string())]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::Error(msg, _) => {
                assert!(msg.contains("requires exactly 1 argument"));
            }
            _ => panic!("Expected error for missing arguments"),
        }
    }

    #[test]
    fn test_select_with_superpose() {
        let env = Environment::new();

        // (select 2 (superpose (A B C D))) - should return first 2 elements
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("select".to_string()),
            MettaValue::Long(2),
            MettaValue::SExpr(vec![
                MettaValue::Atom("superpose".to_string()),
                MettaValue::SExpr(vec![
                    MettaValue::Atom("A".to_string()),
                    MettaValue::Atom("B".to_string()),
                    MettaValue::Atom("C".to_string()),
                    MettaValue::Atom("D".to_string()),
                ]),
            ]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], MettaValue::Atom("A".to_string()));
        assert_eq!(results[1], MettaValue::Atom("B".to_string()));
    }

    #[test]
    fn test_collapse_with_nested_expressions() {
        let mut env = Environment::new();

        // Add rules that return nested S-expressions
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![MettaValue::Atom("pair".to_string())]),
            rhs: MettaValue::SExpr(vec![
                MettaValue::Atom("a".to_string()),
                MettaValue::Long(1),
            ]),
        });
        env.add_rule(Rule {
            lhs: MettaValue::SExpr(vec![MettaValue::Atom("pair".to_string())]),
            rhs: MettaValue::SExpr(vec![
                MettaValue::Atom("b".to_string()),
                MettaValue::Long(2),
            ]),
        });

        // (collapse (pair))
        let value = MettaValue::SExpr(vec![
            MettaValue::Atom("collapse".to_string()),
            MettaValue::SExpr(vec![MettaValue::Atom("pair".to_string())]),
        ]);

        let (results, _) = eval(value, env);
        assert_eq!(results.len(), 1);
        match &results[0] {
            MettaValue::SExpr(elements) => {
                assert_eq!(elements.len(), 2);
                assert!(matches!(elements[0], MettaValue::SExpr(_)));
                assert!(matches!(elements[1], MettaValue::SExpr(_)));
            }
            _ => panic!("Expected S-expression result from collapse"),
        }
    }
}
