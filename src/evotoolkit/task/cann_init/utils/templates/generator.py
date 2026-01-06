# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

from typing import Any, Dict, List, Optional

from .project_json import ProjectJsonGenerator
from .host_tiling import HostTilingGenerator
from .host_operator import HostOperatorGenerator
from .python_bind import PythonBindGenerator
from .model_src import ModelSrcGenerator


class AscendCTemplateGenerator:
    def __init__(self, signature: Dict[str, Any]):
        self.signature = signature
        self._project_json_gen = ProjectJsonGenerator(signature)
        self._host_tiling_gen = HostTilingGenerator(signature)
        self._host_operator_gen = HostOperatorGenerator(signature)
        self._python_bind_gen = PythonBindGenerator(signature)
        self._model_src_gen = ModelSrcGenerator(signature)

    def generate(
        self,
        kernel_src: str,
        tiling_fields: List[Dict[str, str]],
        tiling_func_body: str,
        infer_shape_body: str,
        project_path: str,
        output_alloc_code: Optional[str] = None,
        soc_versions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        host_tiling_src = self._host_tiling_gen.generate(tiling_fields)
        host_operator_src = self._host_operator_gen.generate(
            tiling_func_body=tiling_func_body,
            infer_shape_body=infer_shape_body,
            soc_versions=soc_versions,
        )
        python_bind_src = self._python_bind_gen.generate(output_alloc_code)

        return {
            "project_json_src": self._project_json_gen.generate(),
            "host_tiling_src": host_tiling_src,
            "host_operator_src": host_operator_src,
            "kernel_src": kernel_src,
            "python_bind_src": python_bind_src,
            "model_src": self._model_src_gen.generate(project_path),
        }
