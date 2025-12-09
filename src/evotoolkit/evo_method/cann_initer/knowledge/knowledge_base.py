# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Knowledge Base Implementation for CANNIniter

Two retrieval strategies:
- API: Strict mode (exact match, no guessing)
- Example: Relaxed mode (fuzzy match allowed)
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from ..utils import KnowledgeBase


def _default_index_path() -> str:
    """Get default index path in cache directory"""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    index_dir = cache_dir / "evotoolkit" / "cann_initer"
    index_dir.mkdir(parents=True, exist_ok=True)
    return str(index_dir / "knowledge_index.json")


class KnowledgeBaseConfig:
    """Knowledge base configuration"""

    def __init__(
        self,
        repo_data_path: str = "/root/Huawei_CANN/KernelOptWorkspace/CannOptTask/benchmarks/Repo_Data",
        operator_repos: List[str] = None,
        index_path: str = None,
    ):
        self.repo_data_path = repo_data_path
        self.operator_repos = operator_repos or [
            "ops-nn",
            "ops-transformer",
            "ops-math",
            "ops-cv",
        ]
        self.index_path = index_path or _default_index_path()


class KnowledgeIndexBuilder:
    """Build knowledge index from source directories"""

    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config

    def build_index(self, verbose: bool = True) -> dict:
        """Build complete index"""
        index = {
            "operators": {},
            "apis": [],
            "aliases": {},
            "categories": {},
        }

        # 1. Scan operator repos
        self._scan_operator_repos(index, verbose)

        # 2. Add common APIs
        self._add_common_apis(index)

        # 3. Build aliases
        self._build_aliases(index)

        # 4. Save index
        self._save_index(index)

        return index

    def _scan_operator_repos(self, index: dict, verbose: bool):
        """Scan operator repositories"""
        repo_data = Path(self.config.repo_data_path)
        if not repo_data.exists():
            return

        skip_dirs = {"cmake", "common", "docs", "examples", "experimental",
                     "scripts", "tests", ".git", "build"}

        for repo_name in self.config.operator_repos:
            repo_path = repo_data / repo_name
            if not repo_path.exists():
                continue

            if verbose:
                print(f"  Scanning {repo_name}...")

            for category_dir in repo_path.iterdir():
                if not category_dir.is_dir() or category_dir.name in skip_dirs:
                    continue
                if category_dir.name.startswith("."):
                    continue

                category = category_dir.name
                if category not in index["categories"]:
                    index["categories"][category] = []

                self._scan_category(category_dir, repo_name, category, index)

    def _scan_category(self, path: Path, repo: str, category: str, index: dict):
        """Scan a category directory recursively"""
        skip = {"common", "docs", "cmake", "tests", "experimental"}

        for op_dir in path.iterdir():
            if not op_dir.is_dir() or op_dir.name in skip:
                continue
            if op_dir.name.startswith("."):
                continue

            has_kernel = (op_dir / "op_kernel").exists()
            has_host = (op_dir / "op_host").exists()

            if not has_kernel and not has_host:
                self._scan_category(op_dir, repo, category, index)
                continue

            op_name = op_dir.name
            index["operators"][op_name] = {
                "repo": repo,
                "category": category,
                "path": str(op_dir),
                "has_kernel": has_kernel,
                "has_host": has_host,
            }
            index["categories"][category].append(op_name)

    def _add_common_apis(self, index: dict):
        """Add common Ascend C APIs"""
        index["apis"] = [
            # Vector
            "Add", "Sub", "Mul", "Div", "Muls", "Divs", "Adds", "Subs",
            "Exp", "Log", "Sqrt", "Rsqrt", "Abs", "Neg",
            "ReduceMax", "ReduceMin", "ReduceSum", "ReduceMean",
            "Relu", "Sigmoid", "Tanh", "Gelu", "Silu", "Softmax",
            "Cast", "Duplicate",
            # Cube
            "MatMul", "BatchMatMul", "Gemm",
            # Data Movement
            "DataCopy", "DataCopyPad",
            # Pipeline
            "EnQue", "DeQue", "AllocTensor", "FreeTensor",
            # Buffer
            "LocalTensor", "GlobalTensor", "TPipe", "TQue",
        ]

    def _build_aliases(self, index: dict):
        """Build name aliases (case variants, snake/camel)"""
        aliases = {}
        for op_name in index["operators"]:
            aliases[op_name.lower()] = op_name
            # snake_case -> camelCase
            parts = op_name.split("_")
            camel = parts[0] + "".join(p.capitalize() for p in parts[1:])
            if camel != op_name:
                aliases[camel.lower()] = op_name
        index["aliases"] = aliases

    def _save_index(self, index: dict):
        """Save index to file"""
        with open(self.config.index_path, "w") as f:
            json.dump(index, f, indent=2)


class RealKnowledgeBase(KnowledgeBase):
    """
    Real knowledge base implementation

    - API search: Strict mode (exact match or candidates)
    - Operator search: Relaxed mode (fuzzy match allowed)
    - Auto-rebuild: Automatically rebuild index if missing or empty
    """

    def __init__(self, config: KnowledgeBaseConfig = None, auto_build: bool = True):
        """
        Args:
            config: Knowledge base configuration
            auto_build: If True, automatically build index when missing or empty
        """
        self.config = config or KnowledgeBaseConfig()
        self.auto_build = auto_build
        self.index = self._load_or_build_index()

    def _load_or_build_index(self) -> dict:
        """Load index from file, rebuild if missing or empty"""
        path = Path(self.config.index_path)

        # Try to load existing index
        if path.exists():
            try:
                with open(path) as f:
                    index = json.load(f)
                # Check if index is valid (has operators or apis)
                if index.get("operators") or index.get("apis"):
                    return index
                print(f"[KnowledgeBase] Index file is empty: {path}")
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[KnowledgeBase] Failed to load index: {e}")

        # Auto-build if enabled
        if self.auto_build:
            return self._rebuild_index()

        # Return empty index
        return {"operators": {}, "apis": [], "aliases": {}, "categories": {}}

    def _rebuild_index(self) -> dict:
        """Rebuild the knowledge index"""
        print(f"[KnowledgeBase] Building index from: {self.config.repo_data_path}")
        builder = KnowledgeIndexBuilder(self.config)
        index = builder.build_index(verbose=True)
        print(f"[KnowledgeBase] Index built: {len(index['operators'])} operators, {len(index['apis'])} APIs")
        print(f"[KnowledgeBase] Index saved to: {self.config.index_path}")
        return index

    def rebuild(self) -> None:
        """Force rebuild the index"""
        self.index = self._rebuild_index()

    # =========================================================================
    # API Search (Strict Mode)
    # =========================================================================

    def search_api(self, name: str) -> Dict[str, Any]:
        """
        Strict API search

        Returns:
            {
                "status": "found" | "not_found" | "ambiguous",
                "api_doc": str | None,
                "candidates": list
            }
        """
        # Exact match
        if name in self.index["apis"]:
            return {"status": "found", "api_doc": self._get_api_doc(name), "candidates": []}

        # Case-insensitive match
        name_lower = name.lower()
        for api in self.index["apis"]:
            if api.lower() == name_lower:
                return {"status": "found", "api_doc": self._get_api_doc(api), "candidates": []}

        # Not found - return candidates
        candidates = [api for api in self.index["apis"]
                      if name_lower in api.lower() or api.lower() in name_lower]
        if candidates:
            return {"status": "ambiguous", "api_doc": None, "candidates": candidates[:5]}

        return {"status": "not_found", "api_doc": None, "candidates": []}

    def _get_api_doc(self, name: str) -> str:
        """Get API documentation (placeholder)"""
        # TODO: Load from CANN installation
        return f"[API: {name}] Documentation placeholder"

    # =========================================================================
    # Operator Search (Relaxed Mode)
    # =========================================================================

    def search_operator(self, name: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Relaxed operator search

        Returns:
            {
                "primary": {"name", "repo", "category", "kernel_code", "host_code", "readme"} | None,
                "related": [{"name", "reason"}],
                "confidence": "high" | "medium" | "low"
            }
        """
        # Exact match
        if name in self.index["operators"]:
            return self._build_operator_result(name, "high")

        # Alias match
        name_lower = name.lower()
        if name_lower in self.index["aliases"]:
            canonical = self.index["aliases"][name_lower]
            return self._build_operator_result(canonical, "high")

        # Fuzzy match
        matches = self._fuzzy_match(name, top_k)
        if matches:
            return self._build_operator_result(matches[0], "medium", matches[1:])

        return {"primary": None, "related": [], "confidence": "low"}

    def _build_operator_result(self, name: str, confidence: str, extra_matches: List[str] = None) -> dict:
        """Build operator search result"""
        op_info = self.index["operators"][name]
        return {
            "primary": self._load_operator_code(name, op_info),
            "related": self._find_related(name, op_info, extra_matches),
            "confidence": confidence,
        }

    def _load_operator_code(self, name: str, op_info: dict) -> dict:
        """Load operator source code"""
        op_path = Path(op_info["path"])
        result = {
            "name": name,
            "repo": op_info["repo"],
            "category": op_info["category"],
            "kernel_code": None,
            "host_code": None,
            "readme": None,
        }

        # Load kernel code
        if op_info["has_kernel"]:
            kernel_dir = op_path / "op_kernel"
            cpp_files = list(kernel_dir.glob("*.cpp"))
            if cpp_files:
                result["kernel_code"] = cpp_files[0].read_text(errors="ignore")

        # Load host/tiling code
        if op_info["has_host"]:
            host_dir = op_path / "op_host"
            cpp_files = list(host_dir.glob("*.cpp"))
            tiling_files = [f for f in cpp_files if "tiling" in f.name.lower()]
            if tiling_files:
                result["host_code"] = tiling_files[0].read_text(errors="ignore")
            elif cpp_files:
                result["host_code"] = cpp_files[0].read_text(errors="ignore")

        # Load README
        readme = op_path / "README.md"
        if readme.exists():
            result["readme"] = readme.read_text(errors="ignore")

        return result

    def _find_related(self, name: str, op_info: dict, extra: List[str] = None) -> List[dict]:
        """Find related operators"""
        related = []
        if extra:
            for m in extra:
                related.append({"name": m, "reason": "Similar name"})
        else:
            category = op_info["category"]
            for op in self.index["categories"].get(category, [])[:3]:
                if op != name:
                    related.append({"name": op, "reason": f"Same category: {category}"})
        return related[:3]

    def _fuzzy_match(self, query: str, top_k: int) -> List[str]:
        """Fuzzy match operator names"""
        query_lower = query.lower()
        query_parts = set(re.split(r"[_\s-]", query_lower))

        scores = []
        for op_name in self.index["operators"]:
            op_lower = op_name.lower()
            op_parts = set(re.split(r"[_\s-]", op_lower))

            score = 0
            if query_lower in op_lower or op_lower in query_lower:
                score += 10
            score += len(query_parts & op_parts) * 3
            if op_lower.startswith(query_lower[:3]) if len(query_lower) >= 3 else False:
                score += 2

            if score > 0:
                scores.append((op_name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scores[:top_k]]

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_api_categories(self) -> Dict[str, List[str]]:
        """Get APIs grouped by category"""
        return {
            "vector": [a for a in self.index["apis"] if a in {
                "Add", "Sub", "Mul", "Div", "Exp", "Log", "ReduceMax", "ReduceSum"}],
            "cube": [a for a in self.index["apis"] if a in {"MatMul", "BatchMatMul", "Gemm"}],
            "data": [a for a in self.index["apis"] if a in {"DataCopy", "DataCopyPad"}],
        }

    def get_operator_categories(self) -> Dict[str, List[str]]:
        """Get operators grouped by category"""
        return self.index["categories"]

    def get_available_knowledge_summary(self) -> str:
        """Get summary for LLM progressive disclosure"""
        lines = ["## Available APIs"]
        for cat, apis in self.get_api_categories().items():
            if apis:
                lines.append(f"- {cat}: {', '.join(apis[:8])}")

        lines.append("\n## Available Operator Examples")
        for cat, ops in self.get_operator_categories().items():
            if ops:
                preview = ', '.join(ops[:4])
                lines.append(f"- {cat} ({len(ops)}): {preview}...")

        return "\n".join(lines)
