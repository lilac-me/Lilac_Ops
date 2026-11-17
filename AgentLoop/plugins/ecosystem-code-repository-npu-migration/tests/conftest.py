"""Test bootstrap: recreate the plugin's symlinked skill view.

The canonical skill sources live under ``skills/ecosystem-code-repository-npu-migration/`` and
``plugins/ecosystem-code-repository-npu-migration/skills/`` holds generated links that are gitignored, so
a fresh clone has no skill view until install.sh / install.bat (or this
helper's Python equivalent) creates it.  Importing this module from the test
directory makes ``pytest plugins/ecosystem-code-repository-npu-migration/tests`` self-sufficient.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import plugin_bundle  # noqa: E402

plugin_bundle.ensure_skill_links(plugin_bundle.find_plugin_root(SCRIPTS))
