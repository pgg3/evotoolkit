# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License

"""Knowledge Base Implementation for CANNIniter

Two retrieval strategies:
- API: Strict mode (exact match, no guessing)
- Example: Relaxed mode (fuzzy match allowed)

API scanning from CANN SDK headers with categorization.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from ..utils import KnowledgeBase


def _default_index_path() -> str:
    """Get default index path in cache directory"""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    index_dir = cache_dir / "evotoolkit" / "cann_initer"
    index_dir.mkdir(parents=True, exist_ok=True)
    return str(index_dir / "knowledge_index.json")


def _default_cann_path() -> str:
    """Get default CANN installation path"""
    # Check environment variable first
    cann_path = os.environ.get("ASCEND_HOME_PATH")
    if cann_path and Path(cann_path).exists():
        return cann_path

    # Standard paths
    candidates = [
        "/usr/local/Ascend/ascend-toolkit/latest",
        "/usr/local/Ascend/ascend-toolkit/8.1.RC1",
        "/opt/Ascend/ascend-toolkit/latest",
    ]
    for path in candidates:
        if Path(path).exists():
            return path

    return "/usr/local/Ascend/ascend-toolkit/latest"


def _default_repo_data_path() -> str | None:
    """Get default repo data path from environment or cache

    Search order:
    1. Environment variable EVOTOOLKIT_REPO_DATA
    2. Cache directory ~/.cache/evotoolkit/cann_initer/repo_data
    3. None (operator examples will be unavailable, but API scanning still works)
    """
    # 1. Check environment variable
    env_path = os.environ.get("EVOTOOLKIT_REPO_DATA")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Check cache directory (for future auto-download support)
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cached_path = cache_dir / "evotoolkit" / "cann_initer" / "repo_data"
    if cached_path.exists() and any(cached_path.iterdir()):
        return str(cached_path)

    # 3. No repo data available
    return None


class KnowledgeBaseConfig:
    """Knowledge base configuration"""

    def __init__(
        self,
        repo_data_path: str = None,
        operator_repos: List[str] = None,
        index_path: str = None,
        cann_path: str = None,
    ):
        self.repo_data_path = repo_data_path or _default_repo_data_path()
        self.operator_repos = operator_repos or [
            "ops-nn",
            "ops-transformer",
            "ops-math",
            "ops-cv",
        ]
        self.index_path = index_path or _default_index_path()
        self.cann_path = cann_path or _default_cann_path()


# =============================================================================
# API Category Mapping
# =============================================================================

# Map header file names to API categories
HEADER_TO_CATEGORY = {
    # Vector operations
    "kernel_operator_vec_binary_intf.h": "vec_binary",
    "kernel_operator_vec_binary_scalar_intf.h": "vec_binary_scalar",
    "kernel_operator_vec_unary_intf.h": "vec_unary",
    "kernel_operator_vec_reduce_intf.h": "vec_reduce",
    "kernel_operator_vec_cmpsel_intf.h": "vec_compare",
    "kernel_operator_vec_duplicate_intf.h": "vec_duplicate",
    "kernel_operator_vec_gather_intf.h": "vec_gather",
    "kernel_operator_vec_scatter_intf.h": "vec_scatter",
    "kernel_operator_vec_transpose_intf.h": "vec_transpose",
    "kernel_operator_vec_vconv_intf.h": "vec_convert",
    "kernel_operator_vec_vpadding_intf.h": "vec_padding",
    "kernel_operator_vec_brcb_intf.h": "vec_broadcast",
    "kernel_operator_vec_mulcast_intf.h": "vec_mulcast",
    "kernel_operator_vec_ternary_scalar_intf.h": "vec_ternary",
    "kernel_operator_vec_createvecindex_intf.h": "vec_index",
    "kernel_operator_vec_gather_mask_intf.h": "vec_gather",
    "kernel_operator_vec_bilinearinterpalation_intf.h": "vec_interpolation",
    # Cube/Matrix operations
    "kernel_operator_mm_intf.h": "cube_matmul",
    "kernel_operator_gemm_intf.h": "cube_gemm",
    "kernel_operator_conv2d_intf.h": "cube_conv",
    # Data movement
    "kernel_operator_data_copy_intf.h": "data_copy",
    "kernel_operator_fixpipe_intf.h": "data_fixpipe",
    # Scalar operations
    "kernel_operator_scalar_intf.h": "scalar",
    # Synchronization
    "kernel_operator_determine_compute_sync_intf.h": "sync",
    "kernel_operator_set_atomic_intf.h": "atomic",
    # System
    "kernel_operator_sys_var_intf.h": "system",
    "kernel_operator_common_intf.h": "common",
    # Other
    "kernel_operator_dump_tensor_intf.h": "debug",
    "kernel_operator_list_tensor_intf.h": "tensor_list",
    "kernel_operator_proposal_intf.h": "proposal",
}

# Higher-level category grouping for display
CATEGORY_GROUPS = {
    "Vector Compute": [
        "vec_binary", "vec_unary", "vec_reduce", "vec_compare",
        "vec_ternary", "vec_binary_scalar",
    ],
    "Vector Data": [
        "vec_duplicate", "vec_gather", "vec_scatter", "vec_transpose",
        "vec_convert", "vec_padding", "vec_broadcast", "vec_mulcast",
        "vec_index", "vec_interpolation",
    ],
    "Cube/Matrix": ["cube_matmul", "cube_gemm", "cube_conv"],
    "Data Movement": ["data_copy", "data_fixpipe"],
    "Scalar": ["scalar"],
    "Sync & Atomic": ["sync", "atomic"],
    "System & Debug": ["system", "common", "debug", "tensor_list", "proposal"],
}


class KnowledgeIndexBuilder:
    """Build knowledge index from source directories and CANN headers"""

    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config

    def build_index(self, verbose: bool = True) -> dict:
        """Build complete index"""
        index = {
            "operators": {},
            "apis": {},  # Changed to dict: {api_name: {category, description, header}}
            "api_categories": {},  # {category: [api_names]}
            "aliases": {},
            "categories": {},
        }

        # 1. Scan operator repos
        if verbose:
            print("[IndexBuilder] Scanning operator repositories...")
        self._scan_operator_repos(index, verbose)

        # 2. Scan CANN APIs from headers
        if verbose:
            print("[IndexBuilder] Scanning CANN SDK APIs...")
        self._scan_cann_apis(index, verbose)

        # 3. Build aliases
        self._build_aliases(index)

        # 4. Save index
        self._save_index(index)

        return index

    def _scan_operator_repos(self, index: dict, verbose: bool):
        """Scan operator repositories"""
        if not self.config.repo_data_path:
            if verbose:
                print("  Skipping operator scan: repo_data_path not configured")
                print("  (Set EVOTOOLKIT_REPO_DATA env var or pass repo_data_path to enable)")
            return

        repo_data = Path(self.config.repo_data_path)
        if not repo_data.exists():
            if verbose:
                print(f"  Warning: Repo data path not found: {repo_data}")
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

    def _scan_cann_apis(self, index: dict, verbose: bool):
        """Scan Ascend C APIs from CANN SDK headers"""
        cann_path = Path(self.config.cann_path)

        # Find interface headers directory
        interface_dirs = [
            cann_path / "aarch64-linux" / "ascendc" / "include" / "basic_api" / "interface",
            cann_path / "x86_64-linux" / "ascendc" / "include" / "basic_api" / "interface",
            cann_path / "include" / "ascendc" / "basic_api" / "interface",
        ]

        interface_dir = None
        for d in interface_dirs:
            if d.exists():
                interface_dir = d
                break

        if not interface_dir:
            if verbose:
                print(f"  Warning: CANN interface headers not found in {cann_path}")
            # Fall back to hardcoded common APIs
            self._add_fallback_apis(index)
            return

        if verbose:
            print(f"  Found interface headers: {interface_dir}")

        # Scan each header file
        total_apis = 0
        for header_file in interface_dir.glob("kernel_operator_*.h"):
            if header_file.name.startswith("kernel_struct_"):
                continue

            category = HEADER_TO_CATEGORY.get(header_file.name, "other")
            apis = self._extract_apis_from_header(header_file)

            for api_name, description in apis:
                if api_name not in index["apis"]:
                    index["apis"][api_name] = {
                        "category": category,
                        "description": description,
                        "header": header_file.name,
                    }
                    if category not in index["api_categories"]:
                        index["api_categories"][category] = []
                    index["api_categories"][category].append(api_name)
                    total_apis += 1

        if verbose:
            print(f"  Scanned {total_apis} APIs from {len(index['api_categories'])} categories")

    def _extract_apis_from_header(self, header_path: Path) -> List[tuple]:
        """Extract API function names and descriptions from a header file

        Returns list of (api_name, description) tuples

        Strategy: Find comment blocks with @brief, then find the first function
        declaration after each comment block. This ensures correct matching.
        """
        apis = []
        seen: Set[str] = set()

        try:
            content = header_path.read_text(errors="ignore")
        except Exception:
            return apis

        # Pattern to match comment block followed by function declaration
        # This ensures the @brief belongs to the function that follows it
        pattern = r"""
            /\*[^*]*\*+(?:[^/*][^*]*\*+)*/           # Comment block /* ... */
            \s*                                      # Whitespace
            (?:template\s*<[^>]+>\s*)?               # Optional template
            __aicore__\s+inline\s+                   # __aicore__ inline
            (?:__inout_pipe__\([^)]+\)\s+)?          # Optional pipe annotation
            (?:void|[A-Za-z_][A-Za-z0-9_]*)\s+       # Return type
            ([A-Z][A-Za-z0-9]+)                      # Function name (PascalCase)
            \s*\(                                    # Opening paren
        """

        for match in re.finditer(pattern, content, re.VERBOSE):
            api_name = match.group(1)
            if api_name not in seen:
                seen.add(api_name)
                # Extract @brief from the matched comment block
                comment_block = match.group(0)
                desc = self._extract_description_from_comment(comment_block)
                apis.append((api_name, desc))

        return apis

    def _extract_description_from_comment(self, comment_block: str) -> str:
        """Extract brief description from a comment block"""
        # Find @brief in the comment
        brief_match = re.search(r"@brief\s+(.+?)(?:\n|$)", comment_block)
        if brief_match:
            return brief_match.group(1).strip()
        return ""

    def _add_fallback_apis(self, index: dict):
        """Add common APIs when CANN headers not available"""
        fallback_apis = {
            "vec_binary": ["Add", "Sub", "Mul", "Div", "Max", "Min", "And", "Or"],
            "vec_binary_scalar": ["Adds", "Subs", "Muls", "Divs", "Maxs", "Mins"],
            "vec_unary": ["Abs", "Exp", "Ln", "Sqrt", "Rsqrt", "Relu", "Not", "Reciprocal"],
            "vec_reduce": ["ReduceMax", "ReduceMin", "ReduceSum", "BlockReduceMax",
                          "BlockReduceMin", "BlockReduceSum", "WholeReduceMax",
                          "WholeReduceMin", "WholeReduceSum"],
            "vec_compare": ["Compare", "Select"],
            "vec_duplicate": ["Duplicate"],
            "vec_convert": ["Cast"],
            "cube_matmul": ["Mmad", "LoadData"],
            "cube_gemm": ["Gemm"],
            "data_copy": ["DataCopy", "DataCopyPad", "DataCopyExtParams"],
            "scalar": ["ScalarAdd", "ScalarSub", "ScalarMul", "ScalarDiv"],
            "sync": ["SetFlag", "WaitFlag", "PipeBarrier"],
            "system": ["GetBlockIdx", "GetBlockNum", "GetBlockDim", "pipe_barrier"],
        }

        for category, api_list in fallback_apis.items():
            index["api_categories"][category] = []
            for api_name in api_list:
                index["apis"][api_name] = {
                    "category": category,
                    "description": "",
                    "header": "fallback",
                }
                index["api_categories"][category].append(api_name)

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
        return {"operators": {}, "apis": {}, "api_categories": {}, "aliases": {}, "categories": {}}

    def _rebuild_index(self) -> dict:
        """Rebuild the knowledge index"""
        print("[KnowledgeBase] Building index...")
        print(f"  Operators from: {self.config.repo_data_path or '(not configured)'}")
        print(f"  APIs from: {self.config.cann_path}")
        builder = KnowledgeIndexBuilder(self.config)
        index = builder.build_index(verbose=True)
        api_count = len(index.get("apis", {}))
        print(f"[KnowledgeBase] Index built: {len(index['operators'])} operators, {api_count} APIs")
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
                "api_info": dict | None,
                "candidates": list
            }
        """
        apis = self.index.get("apis", {})

        # Handle both old format (list) and new format (dict)
        if isinstance(apis, list):
            # Old format compatibility
            if name in apis:
                return {"status": "found", "api_info": {"name": name}, "candidates": []}
            name_lower = name.lower()
            for api in apis:
                if api.lower() == name_lower:
                    return {"status": "found", "api_info": {"name": api}, "candidates": []}
            candidates = [api for api in apis
                         if name_lower in api.lower() or api.lower() in name_lower]
            if candidates:
                return {"status": "ambiguous", "api_info": None, "candidates": candidates[:5]}
            return {"status": "not_found", "api_info": None, "candidates": []}

        # New format (dict)
        # Exact match
        if name in apis:
            return {"status": "found", "api_info": {"name": name, **apis[name]}, "candidates": []}

        # Case-insensitive match
        name_lower = name.lower()
        for api_name, api_info in apis.items():
            if api_name.lower() == name_lower:
                return {"status": "found", "api_info": {"name": api_name, **api_info}, "candidates": []}

        # Not found - return candidates
        candidates = [api for api in apis
                      if name_lower in api.lower() or api.lower() in name_lower]
        if candidates:
            return {"status": "ambiguous", "api_info": None, "candidates": candidates[:5]}

        return {"status": "not_found", "api_info": None, "candidates": []}

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
        if name_lower in self.index.get("aliases", {}):
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
    # Utility Methods - Progressive Disclosure
    # =========================================================================

    def get_api_categories(self) -> Dict[str, List[str]]:
        """Get APIs grouped by high-level category"""
        api_categories = self.index.get("api_categories", {})

        # Group by high-level categories
        result = {}
        for group_name, subcategories in CATEGORY_GROUPS.items():
            apis_in_group = []
            for subcat in subcategories:
                apis_in_group.extend(api_categories.get(subcat, []))
            if apis_in_group:
                result[group_name] = apis_in_group

        return result

    def get_operator_categories(self) -> Dict[str, List[str]]:
        """Get operators grouped by category"""
        return self.index.get("categories", {})

    def get_available_knowledge_summary(self) -> str:
        """Get complete knowledge summary for RetrievalPlanner

        Returns a structured overview with:
        - Complete API list (grouped by category)
        - Operator example names (grouped by category)
        """
        lines = []

        # APIs section - COMPLETE list
        lines.append("## Available APIs (Complete List)")
        api_cats = self.get_api_categories()
        if api_cats:
            for cat_name, apis in api_cats.items():
                if apis:
                    # Show ALL APIs in this category
                    api_list = ", ".join(sorted(apis))
                    lines.append(f"- **{cat_name}**: {api_list}")
        else:
            lines.append("- No APIs indexed")

        # Operators section - names only
        lines.append("\n## Available Operator Examples")
        op_cats = self.get_operator_categories()
        if op_cats:
            for cat_name, ops in op_cats.items():
                if ops:
                    # Show all operator names
                    op_list = ", ".join(sorted(ops))
                    lines.append(f"- **{cat_name}**: {op_list}")
        else:
            lines.append("- No operators indexed")

        return "\n".join(lines)

    def get_api_list(self) -> List[str]:
        """Get flat list of all API names"""
        apis = self.index.get("apis", {})
        if isinstance(apis, list):
            return apis
        return list(apis.keys())

    def get_api_count(self) -> int:
        """Get total number of APIs"""
        return len(self.get_api_list())

    def get_operator_count(self) -> int:
        """Get total number of operators"""
        return len(self.index.get("operators", {}))
