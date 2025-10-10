# Architecture

EvoToolkit is designed with modularity and extensibility in mind.

---

## Core Components

### 1. Tasks (`evotoolkit.task`)
Define optimization problems and evaluation logic.

### 2. Methods (`evotoolkit.evo_method`)
Implement evolutionary algorithms (EoH, EvoEngineer, FunSearch).

### 3. Interfaces (`evotoolkit.core.method_interface`)
Bridge between tasks and methods, handling algorithm-specific adaptations.

### 4. Registry (`evotoolkit.registry`)
Automatic discovery and registration of tasks and algorithms.

---

## Design Patterns

- **Factory Pattern**: `evotoolkit.solve()` creates algorithm instances
- **Strategy Pattern**: Interfaces provide algorithm-specific strategies
- **Template Method**: Base classes define workflow, subclasses customize

---

## Module Organization

```
evotool/
├── core/               # Base classes and abstractions
├── evo_method/         # Algorithm implementations
├── task/               # Task implementations
├── tools/              # Utilities (LLM API, etc.)
└── registry.py         # Component registration
```

---

For detailed implementation guides, see the source code documentation.
