"""One-shot headless-cli workspaces, with no cross-call agent sessions."""

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from evotoolkit.core.types import Proposal, ProposalContext
from evotoolkit.evaluators.files import file_path, write_files

from .chat import context_text


class AgentProposer:
    """Run an installed ``headless`` binary and collect its text artifacts.

    ``workdir_factory(ctx)`` must return a context manager yielding a dedicated
    directory. It may supply evaluate.py and support files, which are excluded
    from proposals. Without a factory, a temporary directory is used; an optional
    ``spec.extras['evaluation_command']`` argv creates evaluate.py there.
    Output defaults to the main file and surviving parent files. The agent may
    write ARTIFACTS.json (a list of paths) to declare a different complete file set.

    Headless uses the backend's auto-approval mode. Workspaces are not sandboxes.
    """

    def __init__(self, backend: str, workdir_factory=None, *, timeout: float = 600, eventlog=None):
        if not backend or backend.startswith("-"):
            raise ValueError("backend must name a headless-cli agent")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.backend = backend
        self.workdir_factory = workdir_factory
        self.timeout = timeout
        self.eventlog = eventlog
        self.name = f"agent:{backend}"

    def propose(self, ctx: ProposalContext) -> list[Proposal]:
        if ctx.num_proposals != 1:
            raise ValueError("AgentProposer produces one artifact per call; request num_proposals=1")
        executable = shutil.which("headless")
        if executable is None:
            raise FileNotFoundError("Install headless-cli with npm install -g @roberttlange/headless and authenticate the selected agent")
        call_id = ctx.call_id
        trace = {}
        try:
            workspace = self.workdir_factory(ctx) if self.workdir_factory else TemporaryDirectory(prefix="evotoolkit-agent-")
            with workspace as directory:
                root = Path(directory).resolve()
                root.mkdir(parents=True, exist_ok=True)
                file_path(root, ctx.spec.main_file)
                support = {path.relative_to(root) for path in root.rglob("*") if path.is_file() or path.is_symlink()}
                parent_files = ctx.parents[-1].files if ctx.parents else {}
                reserved = {Path("TASK.md"), Path("evaluate.py"), Path("ARTIFACTS.json")}
                if reserved.intersection(map(Path, parent_files)) or Path(ctx.spec.main_file) in reserved:
                    raise ValueError("TASK.md, evaluate.py and ARTIFACTS.json are reserved workspace files")
                if Path("ARTIFACTS.json") in support:
                    raise ValueError("Workspace factory must not supply ARTIFACTS.json")
                if support.intersection(map(Path, parent_files)) or Path(ctx.spec.main_file) in support:
                    raise ValueError("Parent artifacts would overwrite workspace support files")
                write_files(root, parent_files)
                entry = ctx.spec.extras.get("evaluation_command")
                if entry is not None and Path("evaluate.py") not in support:
                    if not isinstance(entry, list) or not entry or any(not isinstance(arg, str) for arg in entry):
                        raise ValueError("evaluation_command must be a nonempty argv list")
                    write_files(root, {"evaluate.py": f"import subprocess\nraise SystemExit(subprocess.call({entry!r}))\n"})
                prompt = (
                    context_text(ctx)
                    + f"\n\nEdit the artifact files in this directory. The main artifact is {ctx.spec.main_file}. Return one complete candidate."
                )
                prompt += "\nTASK.md, evaluate.py, and existing support files are harness inputs; do not edit them."
                prompt += "\nOnly the main file and surviving parent files are collected by default. To add files, write ARTIFACTS.json as a JSON list of every artifact path, including the main file. Do not include support files or logs."
                prompt += (
                    "\nRun python evaluate.py to check your candidate."
                    if (root / "evaluate.py").exists()
                    else "\nNo local evaluator is configured; evaluation will happen after this call."
                )
                write_files(root, {"TASK.md": prompt})
                command = [
                    executable,
                    self.backend,
                    "--prompt-file",
                    str(root / "TASK.md"),
                    "--work-dir",
                    str(root),
                    "--json",
                    "--usage",
                    "--timeout",
                    str(self.timeout),
                ]
                trace.update(command=command, prompt=prompt)
                result = subprocess.run(
                    command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=self.timeout + 10, check=False
                )
                trace.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
                if result.returncode:
                    raise RuntimeError(f"headless exited {result.returncode}: {result.stderr}")
                names = list(dict.fromkeys([ctx.spec.main_file, *(name for name in parent_files if (root / name).exists())]))
                manifest = file_path(root, "ARTIFACTS.json")
                if manifest.exists():
                    names = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                    raise ValueError("ARTIFACTS.json must contain a list of relative file paths")
                files, paths = {}, set()
                for name in names:
                    path = file_path(root, name)
                    if path in paths or path.relative_to(root) in support | reserved:
                        raise ValueError(f"Duplicate or protected artifact path: {name}")
                    if path.stat().st_nlink != 1:
                        raise ValueError(f"Artifact has hard links: {name}")
                    paths.add(path)
                    files[name] = path.read_text(encoding="utf-8")
                if ctx.spec.main_file not in files:
                    raise ValueError(f"Agent did not produce {ctx.spec.main_file}")
                usage = {}
                for line in result.stdout.splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record, dict) and isinstance(record.get("usage"), dict) and "usageStatus" in record["usage"]:
                        usage = record["usage"]
                return [Proposal(call_id, files, usage=usage)]
        except Exception as exc:
            trace["error"] = str(exc)
            if isinstance(exc, subprocess.TimeoutExpired):
                trace["stdout"] = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout
                trace["stderr"] = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
            raise
        finally:
            if self.eventlog:
                self.eventlog.trace(call_id, trace)
