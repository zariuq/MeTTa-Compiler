//! I/O Operations: file imports and command-line arguments
//!
//! This module implements:
//! - import!: Load and evaluate .metta files
//! - get-args: Get all command-line arguments
//! - get-arg: Get specific command-line argument

use crate::backend::environment::Environment;
use crate::backend::models::{EvalResult, MettaValue};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Import path resolution mode
#[derive(Copy, Clone, Debug)]
enum ImportMode {
    Rel,  // Relative to importing file (default)
    Cwd,  // Relative to current working directory
    Abs,  // Absolute path (explicit)
}

/// Import a .metta file: (import! &space path [mode])
/// Loads and evaluates the file in the specified space
///
/// Examples:
///   (import! &self "helpers.metta")        ; File-relative (default)
///   (import! &self "helpers.metta" rel)    ; File-relative (explicit)
///   (import! &self "exp/lib.metta" cwd)    ; CWD-relative
///   (import! &self "/abs/path.metta" abs)  ; Absolute (explicit)
///
/// Modes:
/// - rel: Relative to importing file's directory (default, portable)
/// - cwd: Relative to current working directory (convenient)
/// - abs: Absolute path (explicit, must start with /)
///
/// Features:
/// - Duplicate detection: Same file won't be loaded twice
/// - Cycle detection: Circular imports produce error
pub(super) fn eval_import(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    // Accept 2 or 3 arguments
    if items.len() < 3 || items.len() > 4 {
        let err = MettaValue::Error(
            format!(
                "import!: expected 2 or 3 arguments, got {}. Usage: (import! &space \"path\" [mode])",
                items.len() - 1
            ),
            Arc::new(MettaValue::SExpr(items[1..].to_vec())),
        );
        return (vec![err], env);
    }

    let space_ref = &items[1];
    let path_value = &items[2];
    let mode_value = items.get(3);

    // Extract space name
    let _space_name = match space_ref {
        MettaValue::Atom(name) if name.starts_with('&') => {
            name.strip_prefix('&').unwrap_or(name)
        }
        _ => {
            let err = MettaValue::Error(
                format!(
                    "import!: first argument must be a space reference like &self or &kb, got: {}",
                    super::friendly_value_repr(space_ref)
                ),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };
    // TODO: Use _space_name for space-targeted imports (import into non-self spaces)

    // Extract path string
    let path_str = match path_value {
        MettaValue::String(s) => s.as_str(),
        MettaValue::Atom(s) => s.as_str(),
        _ => {
            let err = MettaValue::Error(
                format!(
                    "import!: second argument must be a string path, got: {}",
                    super::friendly_value_repr(path_value)
                ),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };

    // Parse import mode (default: rel)
    let mode = if let Some(mode_val) = mode_value {
        match mode_val {
            MettaValue::Atom(s) => match s.as_str() {
                "rel" => ImportMode::Rel,
                "cwd" => ImportMode::Cwd,
                "abs" => ImportMode::Abs,
                other => {
                    let err = MettaValue::Error(
                        format!(
                            "import!: mode must be 'rel', 'cwd', or 'abs', got '{}'",
                            other
                        ),
                        Arc::new(MettaValue::SExpr(items[1..].to_vec())),
                    );
                    return (vec![err], env);
                }
            },
            _ => {
                let err = MettaValue::Error(
                    format!(
                        "import!: mode must be an atom (rel/cwd/abs), got: {}",
                        super::friendly_value_repr(mode_val)
                    ),
                    Arc::new(MettaValue::SExpr(items[1..].to_vec())),
                );
                return (vec![err], env);
            }
        }
    } else {
        ImportMode::Rel  // Default mode
    };

    // Resolve path
    let resolved_path = match resolve_import_path(path_str, &env.current_file, mode) {
        Ok(p) => p,
        Err(e) => {
            let err = MettaValue::Error(
                format!("import!: failed to resolve path '{}': {}", path_str, e),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };

    // Canonicalize for duplicate detection
    let canonical_path = match resolved_path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            let err = MettaValue::Error(
                format!("import!: file not found: '{}' ({})", resolved_path.display(), e),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };

    // Check if already loaded (duplicate import)
    let already_loaded = {
        let loaded = env.loaded_files.read().unwrap();
        loaded.contains(&canonical_path)
    };
    if already_loaded {
        // Already loaded, return () silently
        return (vec![MettaValue::Nil], env);
    }

    // Check for circular import
    let is_circular = {
        let loading = env.loading_stack.read().unwrap();
        loading.contains(&canonical_path)
    };
    if is_circular {
        let err = MettaValue::Error(
            format!(
                "import!: circular import detected: {}",
                canonical_path.display()
            ),
            Arc::new(MettaValue::SExpr(items[1..].to_vec())),
        );
        return (vec![err], env);
    }

    // Load file content
    let content = match fs::read_to_string(&canonical_path) {
        Ok(c) => c,
        Err(e) => {
            let err = MettaValue::Error(
                format!("import!: failed to read file '{}': {}", canonical_path.display(), e),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };

    // Parse file
    let mut parser = match crate::TreeSitterMettaParser::new() {
        Ok(p) => p,
        Err(e) => {
            let err = MettaValue::Error(
                format!("import!: failed to initialize parser: {}", e),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };
    let parsed = match parser.parse(&content) {
        Ok(exprs) => exprs,
        Err(e) => {
            let err = MettaValue::Error(
                format!("import!: parse error in '{}': {}", canonical_path.display(), e),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            return (vec![err], env);
        }
    };

    // Mark as currently loading
    {
        let mut loading = env.loading_stack.write().unwrap();
        loading.push(canonical_path.clone());
    }

    // Update current_file for nested imports
    let old_file = env.current_file.clone();
    let mut new_env = env.clone();
    new_env.current_file = canonical_path.parent().map(|p| p.to_path_buf());

    // Evaluate all expressions in the file
    // If importing into non-self space, we need to handle rules specially
    for expr in parsed {
        // Convert SExpr to MettaValue
        let metta_value = match MettaValue::try_from(&expr) {
            Ok(v) => v,
            Err(e) => {
                let err = MettaValue::Error(
                    format!("import!: conversion error in '{}': {}", canonical_path.display(), e),
                    Arc::new(MettaValue::SExpr(items[1..].to_vec())),
                );
                return (vec![err], new_env);
            }
        };
        let (_, eval_env) = super::eval(metta_value, new_env);
        new_env = eval_env;

        // TODO: If space_name != "self", redirect rule definitions to target space
        // For now, everything goes into the same environment
    }

    // Restore current_file
    new_env.current_file = old_file;

    // Mark as loaded (no longer loading)
    {
        let mut loading = new_env.loading_stack.write().unwrap();
        loading.pop();
        
        let mut loaded = new_env.loaded_files.write().unwrap();
        loaded.insert(canonical_path);
    }

    // Return ()
    (vec![MettaValue::Nil], new_env)
}

/// Resolve import path according to mode
fn resolve_import_path(path_str: &str, current_file: &Option<PathBuf>, mode: ImportMode) -> Result<PathBuf, String> {
    let path = Path::new(path_str);

    // If path is already absolute
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    // Handle mode
    match mode {
        ImportMode::Abs => {
            // User explicitly requested abs mode but path isn't absolute
            Err(format!("mode 'abs' requires absolute path (starting with /), got: {}", path_str))
        }
        ImportMode::Cwd => {
            // Resolve relative to current working directory
            let cwd = std::env::current_dir()
                .map_err(|e| format!("failed to get current directory: {}", e))?;
            Ok(cwd.join(path_str))
        }
        ImportMode::Rel => {
            // Resolve relative to importing file's directory (default)
            if let Some(current) = current_file {
                Ok(current.join(path_str))
            } else {
                // No current file context, fallback to cwd
                let cwd = std::env::current_dir()
                    .map_err(|e| format!("failed to get current directory: {}", e))?;
                Ok(cwd.join(path_str))
            }
        }
    }
}

/// Get all command-line arguments: (get-args)
/// Returns a list of all arguments passed to the program
///
/// Example:
///   # Command: mettatron solver.metta 100 timeout=30
///   !(get-args)
///   ; Returns: ["solver.metta", "100", "timeout=30"]
pub(super) fn eval_get_args(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    if items.len() != 1 {
        let err = MettaValue::Error(
            format!("get-args takes no arguments, got {}", items.len() - 1),
            Arc::new(MettaValue::SExpr(items[1..].to_vec())),
        );
        return (vec![err], env);
    }

    // Convert args to MettaValue list
    let args_list: Vec<MettaValue> = env.cmd_args
        .iter()
        .map(|s: &String| MettaValue::String(s.clone()))
        .collect();

    (vec![MettaValue::SExpr(args_list)], env)
}

/// Get specific command-line argument: (get-arg index) or (get-arg "key")
///
/// Examples:
///   # Command: mettatron solver.metta 100 timeout=30
///   !(get-arg 0)           ; Returns: "solver.metta"
///   !(get-arg 1)           ; Returns: "100"
///   !(get-arg "timeout")   ; Returns: "30" (parses key=value)
///   !(get-arg 99)          ; Returns: Error (index out of bounds)
///   !(get-arg "missing")   ; Returns: Error (key not found)
pub(super) fn eval_get_arg(items: Vec<MettaValue>, env: Environment) -> EvalResult {
    require_args_with_usage!("get-arg", items, 1, env, "(get-arg index) or (get-arg \"key\")");

    let arg_spec = &items[1];

    match arg_spec {
        // Index access: (get-arg 0)
        MettaValue::Long(index) => {
            let idx = *index as usize;
            if idx < env.cmd_args.len() {
                (vec![MettaValue::String(env.cmd_args[idx].clone())], env)
            } else {
                let err = MettaValue::Error(
                    format!(
                        "get-arg: index {} out of bounds (have {} args)",
                        idx,
                        env.cmd_args.len()
                    ),
                    Arc::new(MettaValue::SExpr(items[1..].to_vec())),
                );
                (vec![err], env)
            }
        }
        
        // Key access: (get-arg "timeout")
        MettaValue::String(key) | MettaValue::Atom(key) => {
            // Search for key=value in args
            for arg in &env.cmd_args {
                if let Some((k, v)) = arg.split_once('=') {
                    if k == key.as_str() {
                        return (vec![MettaValue::String(v.to_string())], env);
                    }
                }
            }
            
            let err = MettaValue::Error(
                format!("get-arg: key '{}' not found in arguments", key),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            (vec![err], env)
        }
        
        _ => {
            let err = MettaValue::Error(
                format!(
                    "get-arg: argument must be an index (number) or key (string), got: {}",
                    super::friendly_value_repr(arg_spec)
                ),
                Arc::new(MettaValue::SExpr(items[1..].to_vec())),
            );
            (vec![err], env)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_args_returns_all_args() {
        let mut env = Environment::new();
        env.cmd_args = vec!["100".to_string(), "timeout=30".to_string()];

        let (results, _) = eval_get_args(
            vec![MettaValue::Atom("get-args".to_string())],
            env,
        );

        assert_eq!(results.len(), 1);
        if let MettaValue::SExpr(args) = &results[0] {
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], MettaValue::String("100".to_string()));
            assert_eq!(args[1], MettaValue::String("timeout=30".to_string()));
        } else {
            panic!("Expected SExpr, got {:?}", results[0]);
        }
    }

    #[test]
    fn test_get_arg_by_index() {
        let mut env = Environment::new();
        env.cmd_args = vec!["first".to_string(), "second".to_string()];

        // (get-arg 0)
        let (results, _) = eval_get_arg(
            vec![
                MettaValue::Atom("get-arg".to_string()),
                MettaValue::Long(0),
            ],
            env.clone(),
        );
        assert_eq!(results[0], MettaValue::String("first".to_string()));

        // (get-arg 1)
        let (results, _) = eval_get_arg(
            vec![
                MettaValue::Atom("get-arg".to_string()),
                MettaValue::Long(1),
            ],
            env,
        );
        assert_eq!(results[0], MettaValue::String("second".to_string()));
    }

    #[test]
    fn test_get_arg_by_key() {
        let mut env = Environment::new();
        env.cmd_args = vec!["timeout=30".to_string(), "strategy=discount".to_string()];

        // (get-arg "timeout")
        let (results, _) = eval_get_arg(
            vec![
                MettaValue::Atom("get-arg".to_string()),
                MettaValue::String("timeout".to_string()),
            ],
            env.clone(),
        );
        assert_eq!(results[0], MettaValue::String("30".to_string()));

        // (get-arg "strategy")
        let (results, _) = eval_get_arg(
            vec![
                MettaValue::Atom("get-arg".to_string()),
                MettaValue::String("strategy".to_string()),
            ],
            env,
        );
        assert_eq!(results[0], MettaValue::String("discount".to_string()));
    }

    #[test]
    fn test_get_arg_index_out_of_bounds() {
        let env = Environment::new(); // empty args

        let (results, _) = eval_get_arg(
            vec![
                MettaValue::Atom("get-arg".to_string()),
                MettaValue::Long(0),
            ],
            env,
        );

        assert!(matches!(results[0], MettaValue::Error(_, _)));
    }

    #[test]
    fn test_get_arg_key_not_found() {
        let mut env = Environment::new();
        env.cmd_args = vec!["foo=bar".to_string()];

        let (results, _) = eval_get_arg(
            vec![
                MettaValue::Atom("get-arg".to_string()),
                MettaValue::String("missing".to_string()),
            ],
            env,
        );

        assert!(matches!(results[0], MettaValue::Error(_, _)));
    }

    #[test]
    fn test_resolve_import_path_rel() {
        use std::path::PathBuf;

        let current_file = Some(PathBuf::from("/home/user/project"));
        let result = resolve_import_path("lib.metta", &current_file, ImportMode::Rel).unwrap();
        assert_eq!(result, PathBuf::from("/home/user/project/lib.metta"));
    }

    #[test]
    fn test_resolve_import_path_cwd() {
        let current_dir = std::env::current_dir().unwrap();
        let result = resolve_import_path("lib.metta", &None, ImportMode::Cwd).unwrap();
        assert_eq!(result, current_dir.join("lib.metta"));
    }

    #[test]
    fn test_resolve_import_path_abs() {
        use std::path::PathBuf;

        let result = resolve_import_path("/tmp/lib.metta", &None, ImportMode::Abs).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/lib.metta"));
    }

    #[test]
    fn test_resolve_import_path_abs_rejects_relative() {
        let result = resolve_import_path("lib.metta", &None, ImportMode::Abs);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires absolute path"));
    }

    #[test]
    fn test_resolve_import_path_already_absolute() {
        use std::path::PathBuf;

        // If path is already absolute, mode doesn't matter (except Abs validation)
        let result = resolve_import_path("/tmp/lib.metta", &None, ImportMode::Rel).unwrap();
        assert_eq!(result, PathBuf::from("/tmp/lib.metta"));
    }
}
