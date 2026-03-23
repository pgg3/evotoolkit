# EvoToolkit Core

EvoToolkit is the runtime layer for LLM-driven evolutionary search.

The core package provides:

- built-in methods: `EoH`, `EvoEngineer`, `FunSearch`
- lifecycle bases: `Method`, `IterativeMethod`, `PopulationMethod`
- checkpointing and readable artifacts through `RunStore`
- generic `PythonTask` and `StringTask` SDKs
- generic Python and string interfaces for the built-in methods
- OpenAI-compatible HTTP client helpers in `evotoolkit.tools`

The stable `1.0.2` repository documents only the reusable core. Keep concrete task families, datasets, and hardware-specific workflows in your own package or repository, and build them on top of this runtime.
