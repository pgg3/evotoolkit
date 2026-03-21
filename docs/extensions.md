# Extensions

The core package is intentionally small. Extend it in one of two directions:

- define your own tasks on top of `PythonTask` or `StringTask`
- define your own algorithms on top of `IterativeMethod` or `PopulationMethod`

## Custom Tasks

The recommended workflow for task authors is:

1. Create a `PythonTask` or `StringTask` subclass.
2. Return a `TaskSpec` from `build_python_spec()` or `build_string_spec()`.
3. Implement the evaluation hook for your modality.
4. Reuse a generic interface from `evotoolkit.task`, or provide a custom `MethodInterface`.
5. Expose the task through explicit imports in your own package.

Task registration still exists for advanced integration, but explicit imports are the default workflow.

## Custom Methods

For new algorithms, start from:

- `IterativeMethod` for general step-wise search
- `PopulationMethod` for generation-based population search

Concrete methods own their initialization policy. The runtime does not special-case task initialization beyond carrying `task.spec` into the method state.
