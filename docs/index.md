# EvoToolkit Core

EvoToolkit is the stable runtime layer for LLM-guided evolutionary search.

The core package provides:

- built-in methods: `EoH`, `EvoEngineer`, `FunSearch`
- lifecycle bases: `Method`, `IterativeMethod`, `PopulationMethod`
- checkpointing and readable artifacts through `RunStore`
- generic `PythonTask` and `StringTask` SDKs
- generic interfaces for Python and string optimization

The package is designed to be extended. Keep concrete task families in your own package or repository, and build them on top of this core.
