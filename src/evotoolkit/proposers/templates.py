"""Modality-independent search instructions; output syntax belongs to the proposer."""

from evotoolkit.core.types import Instruction

_DESCRIBE = (
    " First, describe your new algorithm and main steps in one sentence inside boxed {{}}. Next, implement the solution. Do not give additional explanations."
)

INSTRUCTIONS = {
    "i1": Instruction("i1", 0, "Create a new {modality} solution for {main_file}." + _DESCRIBE, "code_block"),
    "e1": Instruction("e1", 2, "Please help me create a new algorithm that has a totally different form from the given ones." + _DESCRIBE, "code_block"),
    "e2": Instruction(
        "e2",
        2,
        "Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them. "
        "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence inside boxed {{}}. Thirdly, implement the solution.",
        "code_block",
    ),
    "m1": Instruction(
        "m1",
        1,
        "Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided." + _DESCRIBE,
        "code_block",
    ),
    "m2": Instruction(
        "m2",
        1,
        "Please identify the main algorithm parameters and assist me in creating a new algorithm that has different parameter settings of the algorithm provided."
        + _DESCRIBE,
        "code_block",
    ),
    "funsearch": Instruction(
        "funsearch", 2, "Propose a new {modality} solution which performs better than the above solutions. Write it to {main_file}.", "code_block"
    ),
    "init": Instruction(
        "init",
        0,
        "Generate a strong first candidate for this task. Use the insights above if relevant as inspiration. Focus on a concrete, correct implementation instead of discussing alternatives.",
        "name_code_thought",
    ),
    "crossover": Instruction(
        "crossover",
        2,
        "Combine the best features from both parent solutions. Use the insights above if relevant as crossover guidance. Create a hybrid solution that combines the strengths of both parents.",
        "name_code_thought",
    ),
    "mutation": Instruction(
        "mutation",
        1,
        "Apply significant changes to the target solution. Use the insights above if relevant as mutation guidance. Create a substantially modified version that explores new optimization directions.",
        "name_code_thought",
    ),
}
