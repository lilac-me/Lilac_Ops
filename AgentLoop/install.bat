@echo off
REM Keep environment changes (for example PLATFORM) local to this script.
setlocal EnableExtensions
REM -----------------------------------------------------------------
REM MindSpeed-Agent one-click install - multi-platform
REM Usage:
REM   install.bat                  Claude Code - default
REM   install.bat claude-code      Claude Code only
REM   install.bat codex            Codex only
REM   install.bat dsh              DSH (DeepSeek Harness) only
REM   install.bat all              All platforms
REM
REM Run from repo root - CMD only. No admin required.
REM
REM High-level flow:
REM   1. validate the requested platform, source Skills, and required CLI;
REM   2. rebuild plugin-local views of the canonical Skill directories;
REM   3. register this checkout as a marketplace and install its plugins.
REM -----------------------------------------------------------------

REM Anchor every relative path to the checkout containing this script. /D also
REM handles the common case where the checkout is on another Windows drive.
cd /d "%~dp0"

REM A platform instruction file acts as a lightweight repository-root marker.
if not exist "CLAUDE.md" if not exist "AGENTS.md" (
    echo ERROR: CLAUDE.md/AGENTS.md not found - script must be in repo root.
    echo   Found: %~dp0
    pause
    exit /b 1
)

REM The no-argument behavior remains Claude Code for backward compatibility.
set "PLATFORM=%~1"
if "%PLATFORM%"=="" set "PLATFORM=claude-code"

REM Units to install: core (default) | all | space-separated unit ids.
REM Install architecture v2: only core by default; domain units on demand.
set "UNIT_ARG=%~2"
if "%UNIT_ARG%"=="" set "UNIT_ARG=core"

REM Reject unknown values before creating links or changing plugin state.
if "%PLATFORM%"=="claude-code" goto :platform_ok
if "%PLATFORM%"=="codex"       goto :platform_ok
if "%PLATFORM%"=="dsh"         goto :platform_ok
if "%PLATFORM%"=="all"         goto :platform_ok
echo ERROR: Unknown platform "%PLATFORM%". Use: claude-code, codex, dsh, all
pause
exit /b 1

:platform_ok

REM Resolve unit list into UNITS. "all" expands to every on-demand unit from
REM catalog/plugins.yaml; "core" yields empty list (core is always installed).
set "UNITS="
if "%UNIT_ARG%"=="all" goto :resolve_all_units
if "%UNIT_ARG%"=="core" goto :units_resolved
set "UNITS=%UNIT_ARG%"
goto :units_resolved

:resolve_all_units
for /f "delims=" %%U in ('python -c "import sys;sys.path.insert(0,'.');from catalog.loader import load_yaml;import pathlib;ps=load_yaml(pathlib.Path('catalog/plugins.yaml')).get('plugins',[]);print(' '.join(p['id'] for p in ps if (p.get('install') or {}).get('unit')!='core'))"') do set "UNITS=%%U"
:units_resolved

echo === MindSpeed-Agent Setup - platform: %PLATFORM% ===
if "%UNITS%"=="" (
    echo === Units: core ===
) else (
    echo === Units: core + %UNITS% ===
)
echo.

REM ---- Preflight: skill directories exist ----------------------------
REM Plugin-local links are regenerated on every run, so all canonical source
REM directories must exist before any old links are removed.
echo [Preflight] Checking skill directories...
set "MISSING="

if not exist "skills\ops-skills\triton\ops-design" (
    echo   MISSING: skills\ops-skills\triton\ops-design
    set "MISSING=1"
)
if not exist "skills\ops-skills\triton\triton-op-dev" (
    echo   MISSING: skills\ops-skills\triton\triton-op-dev
    set "MISSING=1"
)
if not exist "skills\ops-skills\tilelang\tilelang-design" (
    echo   MISSING: skills\ops-skills\tilelang\tilelang-design
    set "MISSING=1"
)
if not exist "skills\ops-skills\tilelang\tilelang-op-dev" (
    echo   MISSING: skills\ops-skills\tilelang\tilelang-op-dev
    set "MISSING=1"
)
if not exist "skills\ops-skills\common_tools\ops-test-gen" (
    echo   MISSING: skills\ops-skills\common_tools\ops-test-gen
    set "MISSING=1"
)
if not exist "skills\ops-skills\common_tools\ops-profiling" (
    echo   MISSING: skills\ops-skills\common_tools\ops-profiling
    set "MISSING=1"
)
if not exist "skills\ops-skills\common_tools\ops-boundary-survey" (
    echo   MISSING: skills\ops-skills\common_tools\ops-boundary-survey
    set "MISSING=1"
)
if not exist "skills\inference-model-skills\model-analysis" (
    echo   MISSING: skills\inference-model-skills\model-analysis
    set "MISSING=1"
)
if not exist "skills\inference-model-skills\deployment" (
    echo   MISSING: skills\inference-model-skills\deployment
    set "MISSING=1"
)
if not exist "skills\inference-model-skills\performance-optimization" (
    echo   MISSING: skills\inference-model-skills\performance-optimization
    set "MISSING=1"
)
if not exist "skills\inference-model-skills\batch-invariance-validation" (
    echo   MISSING: skills\inference-model-skills\batch-invariance-validation
    set "MISSING=1"
)
if not exist "skills\rl-model-skills\alignment\true-on-policy-align" (
    echo   MISSING: skills\rl-model-skills\alignment\true-on-policy-align
    set "MISSING=1"
)
if not exist "skills\doc-skills\common\doc-orchestrator" (
    echo   MISSING: skills\doc-skills\common\doc-orchestrator
    set "MISSING=1"
)
if not exist "skills\doc-skills\generation\doc-design" (
    echo   MISSING: skills\doc-skills\generation\doc-design
    set "MISSING=1"
)
if not exist "skills\doc-skills\generation\doc-generate" (
    echo   MISSING: skills\doc-skills\generation\doc-generate
    set "MISSING=1"
)
if not exist "skills\doc-skills\generation\doc-update" (
    echo   MISSING: skills\doc-skills\generation\doc-update
    set "MISSING=1"
)
if not exist "skills\doc-skills\review\doc-quality-check" (
    echo   MISSING: skills\doc-skills\review\doc-quality-check
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\model-adaptation\goal" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\model-adaptation\goal
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\model-adaptation\repair" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\model-adaptation\repair
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\model-analysis\feature-model" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\model-analysis\feature-model
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\deployment\env-setup" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\deployment\env-setup
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\deployment\vllm-ascend-env" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\deployment\vllm-ascend-env
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\common\local-assets" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\common\local-assets
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\common\report-pr" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\common\report-pr
    set "MISSING=1"
)
if not exist "skills\ecosystem-code-repository-npu-migration\common\docs-support" (
    echo   MISSING: skills\ecosystem-code-repository-npu-migration\common\docs-support
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-mfu-calculation" (
    echo   MISSING: skills\veomni-skills\veomni-mfu-calculation
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-model-adaptation" (
    echo   MISSING: skills\veomni-skills\veomni-model-adaptation
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-model-analysis" (
    echo   MISSING: skills\veomni-skills\veomni-model-analysis
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-npu-environment" (
    echo   MISSING: skills\veomni-skills\veomni-npu-environment
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-parallel-recipe" (
    echo   MISSING: skills\veomni-skills\veomni-parallel-recipe
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-performance-tuning" (
    echo   MISSING: skills\veomni-skills\veomni-performance-tuning
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-precision-alignment" (
    echo   MISSING: skills\veomni-skills\veomni-precision-alignment
    set "MISSING=1"
)
if not exist "skills\veomni-skills\veomni-quick-migration" (
    echo   MISSING: skills\veomni-skills\veomni-quick-migration
    set "MISSING=1"
)

if defined MISSING (
    echo   One or more skill directories missing. Full clone required.
    pause
    exit /b 1
)
echo   [OK] Skill directories

REM ---- Preflight: platform CLIs ------------------------------------
REM Check only the CLIs needed by the selected mode. "all" checks both before
REM installation begins so it cannot fail halfway due to a missing executable.
if "%PLATFORM%"=="claude-code" goto :check_claude_cli
if "%PLATFORM%"=="all"         goto :check_claude_cli
if "%PLATFORM%"=="dsh"         goto :check_dsh_cli
goto :check_codex_cli

:check_claude_cli
call claude --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: "claude" CLI not found in PATH.
    echo   Install Claude Code: https://code.claude.com/docs/en/setup
    pause
    exit /b 1
)
if "%PLATFORM%"=="claude-code" goto :cli_ok
REM "all" falls through to the dsh check.

:check_dsh_cli
call dsh --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: "dsh" CLI not found in PATH.
    echo   Install DeepSeek Harness: npm install -g @deepseek-ai/dsh
    pause
    exit /b 1
)
if "%PLATFORM%"=="dsh" goto :cli_ok
REM "all" falls through to the codex check.

:check_codex_cli
call codex --version >nul 2>nul
if errorlevel 1 (
    echo ERROR: "codex" CLI not found in PATH.
    pause
    exit /b 1
)

:cli_ok
echo.

REM =================================================================
REM Step 1/2: Create skill junctions
REM =================================================================
echo [1/2] Creating skill junctions...

REM Manifests reference Skills inside each plugin. These directories are only
REM aggregation points; canonical Skill content remains under skills\*.
if not exist "plugins\ops\skills"                      mkdir "plugins\ops\skills"
if not exist "plugins\inference-model-optimize\skills" mkdir "plugins\inference-model-optimize\skills"
if not exist "plugins\doc-agent\skills"                mkdir "plugins\doc-agent\skills"
if not exist "plugins\mindspeed-llm\skills"            mkdir "plugins\mindspeed-llm\skills"

REM Recreate generated views for every requested unit (idempotent). Each call
REM rebuilds plugins\<unit>\skills\* junctions from canonical skills sources.
REM core (mindspeed-agent) reads ./skills/common directly, so no view needed.
if not defined UNITS goto :step1_done
for %%U in (%UNITS%) do (
    echo   Building skill views for unit: %%U
    python "tools\install-unit.py" link %%U
    if errorlevel 1 (echo   FAILED: link views for %%U & pause & exit /b 1)
)
:step1_done
echo   Skill views OK.

REM =================================================================
REM DSH: flat skill projection into .dsh\skills
REM =================================================================
REM DSH discovers skills from <projectRoot>\.dsh\skills\<name>\SKILL.md
REM (single level only, no nested discovery). Project every existing
REM skills\**\SKILL.md directory into a flat junction under .dsh\skills\<name>.
REM Generated artifact (gitignored), rebuilt wholesale on every run; the rule
REM is deterministic and matches the folder reference in .dsh-plugin\plugin.json.
if "%PLATFORM%"=="dsh" goto :dsh_project
if "%PLATFORM%"=="all" goto :dsh_project
goto :skip_dsh

:dsh_project
echo [DSH] Projecting skills into .dsh\skills ...
if exist ".dsh\skills" rmdir /S /Q ".dsh\skills" 2>nul
mkdir ".dsh\skills" 2>nul
REM core (mindspeed-agent) is always projected first; domain units follow.
python "tools\install-unit.py" dsh mindspeed-agent
if errorlevel 1 (echo   FAILED: dsh projection for mindspeed-agent & pause & exit /b 1)
if not defined UNITS goto :dsh_done
for %%U in (%UNITS%) do (
    python "tools\install-unit.py" dsh %%U
    if errorlevel 1 (echo   FAILED: dsh projection for %%U & pause & exit /b 1)
)
:dsh_done
echo.

:skip_dsh

REM =================================================================
REM Step 2/2: Install plugins
REM =================================================================
echo.
echo [2/2] Installing plugins for: %PLATFORM%

REM "all" intentionally enters the Claude branch first, then falls through to
REM the Codex branch after Claude installation completes.
if "%PLATFORM%"=="claude-code" goto :install_claude
if "%PLATFORM%"=="all"         goto :install_claude
if "%PLATFORM%"=="dsh"         goto :done
goto :check_codex

:install_claude
echo   --- Claude Code ---

REM Register the current checkout as the MindSpeed-Agent marketplace. Do not
REM silently replace an existing registration that points to another checkout.
echo   Adding marketplace...
call claude plugin marketplace add "%CD%"
if errorlevel 1 (
    echo ERROR: Failed to add the MindSpeed-Agent marketplace for Claude Code.
    echo   An existing marketplace may point to another checkout; resolve it explicitly and retry.
    pause
    exit /b 1
)

REM core is always installed first.
echo   Installing core (mindspeed-agent)...
call claude plugin install mindspeed-agent@MindSpeed-Agent

REM Install requested on-demand units. Doc-Agent needs junction materialization
REM (see :stage_doc_skills_for_install) because Windows installers do not follow
REM junctions reliably; stage before and restore after only when doc-agent is in
REM the unit list.
for %%U in (%UNITS%) do (
    if "%%U"=="doc-agent" call :stage_doc_skills_for_install
    echo   Installing %%U...
    if "%%U"=="mindspeed-scan-issue-agent" (
        python tools/install-unit.py claude %%U --run
    ) else (
        call claude plugin install %%U@MindSpeed-Agent
    )
    if errorlevel 1 (
        if "%%U"=="doc-agent" call :restore_doc_skill_junctions
        echo ERROR: Claude Code %%U installation failed.
        pause
        exit /b 1
    )
    if "%%U"=="doc-agent" call :restore_doc_skill_junctions
)

echo   Installing veomni-agent...
call claude plugin install veomni-agent@MindSpeed-Agent

echo   Claude Code done.
echo.
if "%PLATFORM%"=="claude-code" goto :done

:check_codex
REM This guard also documents the only routes into the Codex install branch.
if "%PLATFORM%"=="codex" goto :install_codex
if "%PLATFORM%"=="all"   goto :install_codex
echo   ERROR: Unknown platform "%PLATFORM%". Use: claude-code, codex, all
pause
exit /b 1

:install_codex
echo   --- Codex ---
if exist ".codex-plugin\plugin.json" (echo   .codex-plugin/plugin.json found) else (echo   WARN: .codex-plugin/plugin.json not found)
if exist "AGENTS.md" (echo   AGENTS.md found) else (echo   WARN: AGENTS.md not found)

REM Register the same checkout for Codex. Existing registrations are left for
REM the user to resolve explicitly instead of being removed or refreshed here.
echo   Adding marketplace...
call codex plugin marketplace add "%CD%"
if errorlevel 1 (
    echo ERROR: Failed to add the MindSpeed-Agent marketplace for Codex.
    echo   An existing marketplace may point to another checkout; resolve it explicitly and retry.
    pause
    exit /b 1
)

REM core is always installed first.
echo   Installing core (mindspeed-agent)...
call codex plugin add mindspeed-agent@MindSpeed-Agent

REM Install requested on-demand units (doc-agent staged/restored around install).
for %%U in (%UNITS%) do (
    if "%%U"=="doc-agent" call :stage_doc_skills_for_install
    echo   Installing %%U...
    if "%%U"=="mindspeed-scan-issue-agent" (
        python tools/install-unit.py codex %%U --run
    ) else (
        call codex plugin add %%U@MindSpeed-Agent
    )
    if errorlevel 1 (
        if "%%U"=="doc-agent" call :restore_doc_skill_junctions
        echo ERROR: Codex %%U installation failed.
        pause
        exit /b 1
    )
    if "%%U"=="doc-agent" call :restore_doc_skill_junctions
)

echo   Codex done.
echo.

REM =================================================================
REM Common successful exit for one-platform and all-platform modes.
:done
echo === Done. ===
echo.
if "%PLATFORM%"=="claude-code" echo Claude Code: launch and describe your task
if "%PLATFORM%"=="codex"       echo Codex:    launch codex . in this directory
if "%PLATFORM%"=="dsh"         echo DSH:       launch dsh web in this directory
if "%PLATFORM%"=="all"         echo Claude Code: launch and describe your task
if "%PLATFORM%"=="all"         echo Codex:    launch codex . in this directory
if "%PLATFORM%"=="all"         echo DSH:       launch dsh web in this directory

REM Stop the main flow here. The labels below are helper subroutines and must
REM only be entered through CALL.
exit /b 0

REM =================================================================
REM Windows packaging helpers (Doc-Agent and ecosystem-code-repository-npu-migration)
REM
REM During development, plugins\<name>\skills contains junctions to the
REM canonical skills\* sources, so each Skill has only one maintained
REM copy. Windows plugin installers copy the plugin into a cache without
REM following those junctions reliably. The install flow therefore:
REM   1. replaces the junctions with temporary real-directory copies;
REM   2. installs the plugin while those copies are present;
REM   3. restores the junctions for continued development.
REM =================================================================

:remove_plugin_skill
REM A plain RMDIR removes a junction without touching its target. If the path
REM remains, it is a staged real directory and must be removed recursively.
rmdir "plugins\%~1\skills\%~2" 2>nul
if exist "plugins\%~1\skills\%~2" rmdir /S /Q "plugins\%~1\skills\%~2"
if exist "plugins\%~1\skills\%~2" exit /b 1
exit /b 0

:copy_plugin_skill
REM Copy one canonical Skill into the plugin as real files, then verify the
REM required SKILL.md entry point exists before installation continues.
xcopy "%~1" "plugins\%~2\skills\%~3\" /E /I /H /Q /Y >nul
if errorlevel 1 exit /b 1
if not exist "plugins\%~2\skills\%~3\SKILL.md" exit /b 1
exit /b 0

:stage_doc_skills_for_install
REM Replace all five development junctions with real directories so the plugin
REM cache receives the complete Skill contents.
echo   Staging Doc-Agent skills as real directories for plugin installation...
for %%S in (doc-orchestrator doc-design doc-generate doc-update doc-quality-check) do (
    call :remove_plugin_skill "doc-agent" "%%S"
    if errorlevel 1 exit /b 1
)
call :copy_plugin_skill "skills\doc-skills\common\doc-orchestrator" "doc-agent" "doc-orchestrator"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\doc-skills\generation\doc-design" "doc-agent" "doc-design"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\doc-skills\generation\doc-generate" "doc-agent" "doc-generate"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\doc-skills\generation\doc-update" "doc-agent" "doc-update"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\doc-skills\review\doc-quality-check" "doc-agent" "doc-quality-check"
if errorlevel 1 exit /b 1
echo   Doc-Agent skills staged for plugin installation.
exit /b 0

:restore_doc_skill_junctions
REM Remove the temporary copies and return the source tree to its normal
REM junction-based layout. This prevents duplicate maintained Skill copies.
for %%S in (doc-orchestrator doc-design doc-generate doc-update doc-quality-check) do (
    call :remove_plugin_skill "doc-agent" "%%S"
    if errorlevel 1 exit /b 1
)
mklink /J "plugins\doc-agent\skills\doc-orchestrator" "skills\doc-skills\common\doc-orchestrator" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\doc-agent\skills\doc-design" "skills\doc-skills\generation\doc-design" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\doc-agent\skills\doc-generate" "skills\doc-skills\generation\doc-generate" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\doc-agent\skills\doc-update" "skills\doc-skills\generation\doc-update" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\doc-agent\skills\doc-quality-check" "skills\doc-skills\review\doc-quality-check" >nul 2>nul
if errorlevel 1 exit /b 1
exit /b 0

:stage_npu_skills_for_install
REM Replace the eight ecosystem-code-repository-npu-migration development junctions with real directories
REM so the Codex plugin cache receives the complete Skill contents.
echo   Staging ecosystem-code-repository-npu-migration skills as real directories for plugin installation...
for %%S in (goal repair feature-model env-setup vllm-ascend-env local-assets report-pr docs-support) do (
    call :remove_plugin_skill "ecosystem-code-repository-npu-migration" "%%S"
    if errorlevel 1 exit /b 1
)
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\model-adaptation\goal" "ecosystem-code-repository-npu-migration" "goal"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\model-adaptation\repair" "ecosystem-code-repository-npu-migration" "repair"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\model-analysis\feature-model" "ecosystem-code-repository-npu-migration" "feature-model"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\deployment\env-setup" "ecosystem-code-repository-npu-migration" "env-setup"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\deployment\vllm-ascend-env" "ecosystem-code-repository-npu-migration" "vllm-ascend-env"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\common\local-assets" "ecosystem-code-repository-npu-migration" "local-assets"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\common\report-pr" "ecosystem-code-repository-npu-migration" "report-pr"
if errorlevel 1 exit /b 1
call :copy_plugin_skill "skills\ecosystem-code-repository-npu-migration\common\docs-support" "ecosystem-code-repository-npu-migration" "docs-support"
if errorlevel 1 exit /b 1
echo   ecosystem-code-repository-npu-migration skills staged for plugin installation.
exit /b 0

:restore_npu_skill_junctions
REM Remove the temporary copies and return the ecosystem-code-repository-npu-migration plugin view to its
REM junction-based layout pointing at the canonical domain sources.
for %%S in (goal repair feature-model env-setup vllm-ascend-env local-assets report-pr docs-support) do (
    call :remove_plugin_skill "ecosystem-code-repository-npu-migration" "%%S"
    if errorlevel 1 exit /b 1
)
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\goal" "skills\ecosystem-code-repository-npu-migration\model-adaptation\goal" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\repair" "skills\ecosystem-code-repository-npu-migration\model-adaptation\repair" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\feature-model" "skills\ecosystem-code-repository-npu-migration\model-analysis\feature-model" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\env-setup" "skills\ecosystem-code-repository-npu-migration\deployment\env-setup" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\vllm-ascend-env" "skills\ecosystem-code-repository-npu-migration\deployment\vllm-ascend-env" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\local-assets" "skills\ecosystem-code-repository-npu-migration\common\local-assets" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\report-pr" "skills\ecosystem-code-repository-npu-migration\common\report-pr" >nul 2>nul
if errorlevel 1 exit /b 1
mklink /J "plugins\ecosystem-code-repository-npu-migration\skills\docs-support" "skills\ecosystem-code-repository-npu-migration\common\docs-support" >nul 2>nul
if errorlevel 1 exit /b 1
exit /b 0
