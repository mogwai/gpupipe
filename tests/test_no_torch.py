"""PyTorch is optional: CPU pipelines run without it, and their parent and
worker processes never import it. Each case runs a small script in a fresh
interpreter, so nothing this test session imported leaks in."""
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("pipe._rustq")

SCRIPT = textwrap.dedent('''
    import sys

    import numpy as np

    from pipe import Pipe


    class Source:
        def __call__(self):
            for i in range(50):
                yield {"id": i, "wave": np.full(1 << 16, i, dtype=np.float32)}


    class Check:
        def __call__(self, item):
            item["ok"] = bool((item["wave"] == item["id"]).all())
            item["torch_in_worker"] = "torch" in sys.modules
            return item


    class ThreadCheck(Check):
        pass


    if __name__ == "__main__":
        pipe = Pipe(stats_interval=0)
        pipe.add(Source(), outqn=8)
        pipe.add(Check(), workers=2, outqn=8)
        pipe.add(ThreadCheck(), workers=2, thread=True, outqn=8)
        got = list(pipe)
        assert sorted(x["id"] for x in got) == list(range(50)), len(got)
        assert all(x["ok"] for x in got)
        print("WORKERS_IMPORTED_TORCH", any(x["torch_in_worker"] for x in got))
        print("PARENT_IMPORTED_TORCH", "torch" in sys.modules)
''')


def _run(tmp_path, extra_path=None):
    script = tmp_path / "cpu_pipeline.py"
    script.write_text(SCRIPT)
    env = dict(os.environ, PIPE_DRAIN_GRACE="0.2")
    if extra_path:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(extra_path), env.get("PYTHONPATH")]))
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_cpu_pipeline_never_imports_torch(tmp_path):
    out = _run(tmp_path)
    assert "WORKERS_IMPORTED_TORCH False" in out
    assert "PARENT_IMPORTED_TORCH False" in out


def test_cpu_pipeline_runs_when_torch_is_not_installed(tmp_path):
    """A `torch` that fails to import stands in for a torch-free install, in
    the parent and in every worker (they inherit PYTHONPATH)."""
    fake = tmp_path / "no_torch" / "torch"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text('raise ImportError("torch is not installed")\n')
    out = _run(tmp_path, extra_path=fake.parent)
    assert "PARENT_IMPORTED_TORCH False" in out


def test_torch_only_features_say_what_to_install(monkeypatch):
    from pipe import _torch

    monkeypatch.setitem(sys.modules, "torch", None)  # import now fails
    assert _torch.available() is None
    with pytest.raises(ImportError, match=r"gpupipe\[torch\]"):
        _torch.required("pipe.web")
