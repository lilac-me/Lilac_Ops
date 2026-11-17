"""Build a standalone plugin from canonical skills; never sync a tracked copy."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

PLUGIN = Path(__file__).resolve().parents[1]
SOURCE = PLUGIN.parents[1] / "skills/common/common/gitcode-issue-triage"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def build_package(destination: Path) -> Path:
    destination = destination.resolve()
    if destination == PLUGIN or PLUGIN in destination.parents:
        raise ValueError("package output must be outside the source plugin")
    if destination.exists():
        raise FileExistsError(destination)
    if not (SOURCE / "SKILL.md").is_file():
        raise FileNotFoundError("canonical skill missing; build from the repository checkout")
    # Ignore the development view, including missing/stale links. Package only
    # the canonical source so a previous local install cannot ship stale skills.
    def ignore(directory, names):
        excluded = set(IGNORE(directory, names))
        if Path(directory) == PLUGIN:
            excluded.add("skills")
        return excluded
    shutil.copytree(PLUGIN, destination, ignore=ignore)
    shutil.copytree(SOURCE, destination / "skills/gitcode-issue-triage", ignore=IGNORE)
    return destination


@contextmanager
def materialized_install():
    """Expose staged real files to marketplace installers, then restore the view.

    Refuse an existing backup rather than overwriting an interrupted install.
    The temporary directory is owned by this call; canonical files are untouched.
    """
    view = PLUGIN / "skills"
    backup = PLUGIN / ".skills-install-backup"
    if backup.exists() or backup.is_symlink():
        raise FileExistsError(f"previous install backup needs recovery: {backup}")
    with tempfile.TemporaryDirectory(prefix="mindspeed-issue-package-") as tmp:
        package = build_package(Path(tmp) / PLUGIN.name)
        had_view = view.exists()
        if had_view:
            view.rename(backup)
        try:
            shutil.copytree(package / "skills", view)
            yield package
        finally:
            # This directory was created above by copytree, never a junction.
            if view.exists():
                shutil.rmtree(view)
            if had_view:
                backup.rename(view)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path,
                        help="new standalone plugin directory (must not exist)")
    args = parser.parse_args()
    print(build_package(args.output))


if __name__ == "__main__":
    main()
