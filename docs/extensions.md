# Extensions

The core package intentionally does not ship concrete task families anymore.

You have two extension paths:

- install `evotoolkit-tasks` and import a domain package directly
- implement your own `PythonTask` or `StringTask` subclass and pair it with the generic interfaces in `evotoolkit.task`

Task registration remains available for extension authors through `evotoolkit.registry.register_task`, but explicit imports are the primary workflow.
