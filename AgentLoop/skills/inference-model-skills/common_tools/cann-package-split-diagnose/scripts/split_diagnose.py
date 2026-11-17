# -*- coding: utf-8 -*-
"""
CANN 拆包定位辅助脚本 - 交互式生成安装/启动/替换/清理脚本

用法：
    python split_diagnose.py

说明：
  - 所有服务器、路径信息通过交互式输入获取，无硬编码。
  - 不直接执行远程命令，而是先在本地生成一组 bash 脚本，再上传到服务器。
  - 生成的脚本按阶段组织：good/bad 整包安装、组合安装、组件替换、启动、清理。
  - 中文注释与报错提示，方便在终端直接阅读。

重要经验（详见 SKILL.md）：
  - 安装顺序必须 toolkit -> nnal -> ops，并依次 source set_env.sh。
  - 组合环境必须独立全新安装，严禁 cp -a 复制（会保留软链/硬编码绝对路径并污染 good）。
  - 子组件替换必须用 --full --install-path=<path>（--install 会 RC=1 + Usage）。
  - 判定装好要看日志 RC=0 + ALL_DONE + 关键产物存在（libhcomm.so / libhccl.so）。
  - 清理 NPU 需 kill APIServer + EngineCore + VLLMWorker，再 npu-smi info 确认。

阶段：
  Phase 1: 生成 good / bad 整包安装脚本
  Phase 2: 生成组合安装脚本（good tk + good nnal + bad ops）
  Phase 3: 生成 ops 拆包脚本（--noexec --extract）
  Phase 4: 生成组件替换脚本（--full）
  Phase 5: 生成 vLLM 启动脚本 + 清理脚本
  Phase 6: 输出报告模板
"""

import os
import sys

VERSION = "1.0.0"

class Config:
    """交互式收集所有连接与路径信息。"""

    def __init__(self):
        self.host = ""
        self.port = 22
        self.user = "root"
        self.password = ""
        self.good_dir = ""      # 本地 good 包目录（含 3 个 .run）
        self.bad_dir = ""       # 本地 bad 包目录（含 3 个 .run）
        self.workdir = ""       # 服务器工作目录 (如 /mnt/share/<user>/agent-skill)
        self.install_root = ""  # 服务器安装根目录 (如 .../installs)
        self.python_path = "/usr/local/python3.12.13/bin"
        self.vllm_dir = ""      # vLLM 源码目录
        self.port_num = "8026"
        self.ops_type = "A3-ops"
        self.cann_version = "9.1.0"
        self.start_command = ""
        self.verify_cmd = ""

    def prompt(self):
        print("=" * 64)
        print("  CANN 拆包定位 - 脚本生成器")
        print("=" * 64)
        self.host = input("服务器 IP: ").strip()
        p = input("SSH 端口 (默认 22): ").strip()
        if p:
            self.port = int(p)
        u = input("SSH 用户 (默认 root): ").strip()
        if u:
            self.user = u
        import getpass
        self.password = getpass.getpass("SSH 密码: ").strip()

        print()
        self.good_dir = input("本地 good 包目录: ").strip()
        self.bad_dir = input("本地 bad 包目录: ").strip()

        print()
        wd = input("服务器工作目录 (如 /mnt/share/<user>/agent-skill): ").strip()
        if wd:
            self.workdir = wd
        ir = input(f"服务器安装根目录 (默认 {self.workdir}/installs): ").strip()
        if ir:
            self.install_root = ir
        else:
            self.install_root = f"{self.workdir}/installs"

        pp = input(f"Python bin 路径 (默认 {self.python_path}): ").strip()
        if pp:
            self.python_path = pp
        vd = input("vLLM 源码目录: ").strip()
        if vd:
            self.vllm_dir = vd
        pn = input(f"vLLM 端口 (默认 8026): ").strip()
        if pn:
            self.port_num = pn
        ot = input(f"ops 包类型 (默认 A3-ops): ").strip()
        if ot:
            self.ops_type = ot
        cv = input(f"CANN 版本 (默认 9.1.0): ").strip()
        if cv:
            self.cann_version = cv

        print()
        print("请输入 vLLM 启动命令（多行，空行结束；仅第一行写入脚本并追加端口）：")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        if lines:
            self.start_command = "\n".join(lines)

        print()
        print("请输入每轮判定的验证命令（用于判断该组合是正常还是异常）。")
        print("  - 它可能是接口请求、日志/报错抓取、性能指标等，由用户场景决定，不一定相同。")
        vc = input("验证命令（直接回车则使用下面这个 vLLM 示例）: ").strip()
        if vc:
            self.verify_cmd = vc
        else:
            # 仅作示意，不是固定标准；用户场景不同时应自行提供
            self.verify_cmd = (
                f'curl -s http://localhost:{self.port_num}/v1/completions '
                f'-H "Content-Type: application/json" '
                f'-d \'{{"model": "", "prompt": "San Francisco is a", "max_tokens": "20"}}\''
            )

    def summary(self):
        return (
            f"服务器     : {self.user}@{self.host}:{self.port}\n"
            f"good 包目录 : {self.good_dir}\n"
            f"bad 包目录  : {self.bad_dir}\n"
            f"工作目录    : {self.workdir}\n"
            f"安装根目录  : {self.install_root}\n"
            f"vLLM 目录   : {self.vllm_dir}\n"
            f"端口        : {self.port_num}\n"
            f"Ops/版本    : {self.ops_type} / {self.cann_version}"
        )

    def pkg(self, kind, sub):
        """构造某个子包 .run 的路径。kind: good/bad, sub: toolkit/nnal/<ops_type>-ops"""
        d = self.good_dir if kind == "good" else self.bad_dir
        if sub == "toolkit":
            return f"{d}/Ascend-cann-toolkit_{self.cann_version}_linux-aarch64.run"
        if sub == "nnal":
            return f"{d}/Ascend-cann-nnal_{self.cann_version}_linux-aarch64.run"
        return f"{d}/Ascend-cann-{sub}_{self.cann_version}_linux-aarch64.run"


def gen_full_install(cfg, kind, inst):
    """整包安装脚本：toolkit -> nnal -> ops，顺序 source 环境，打印 RC/ALL_DONE。"""
    pk = cfg.pkg(kind, "toolkit")
    pn = cfg.pkg(kind, "nnal")
    po = cfg.pkg(kind, cfg.ops_type)
    inst_path = cfg.install_root + "/" + inst
    lines = [
        "#!/bin/bash",
        "set -x",
        f"export PATH={cfg.python_path}:$PATH",
        f"INST={inst_path}",
        "rm -rf $INST",
        "mkdir -p $INST",
        "",
        "# 1) toolkit",
        f"{pk} --install --install-path=$INST --quiet 2>&1",
        "echo TK_RC=$?",
        "source $INST/cann/set_env.sh",
        "",
        "# 2) nnal",
        f"{pn} --install --install-path=$INST --quiet 2>&1",
        "echo NNAL_RC=$?",
        "source $INST/nnal/atb/set_env.sh",
        "",
        "# 3) ops",
        f"{po} --install --install-path=$INST --quiet 2>&1",
        "echo OPS_RC=$?",
        "",
        "echo ALL_DONE",
    ]
    return "\n".join(lines)

def gen_combo_install(cfg, inst="combo1"):
    """组合安装：good tk + good nnal + bad ops，装到独立目录 (严禁 cp -a)。"""
    pk = cfg.pkg("good", "toolkit")
    pn = cfg.pkg("good", "nnal")
    po = cfg.pkg("bad", cfg.ops_type)
    inst_path = cfg.install_root + "/" + inst
    lines = [
        "#!/bin/bash",
        "set -x",
        f"export PATH={cfg.python_path}:$PATH",
        f"INST={inst_path}",
        "rm -rf $INST",
        "mkdir -p $INST",
        "",
        "# 组合 = good tk + good nnal + bad ops (独立全新安装，严禁 cp -a)",
        f"{pk} --install --install-path=$INST --quiet 2>&1",
        "echo TK_RC=$?",
        "source $INST/cann/set_env.sh",
        f"{pn} --install --install-path=$INST --quiet 2>&1",
        "echo NNAL_RC=$?",
        "source $INST/nnal/atb/set_env.sh",
        f"{po} --install --install-path=$INST --quiet 2>&1",
        "echo OPS_RC=$?",
        "",
        "echo ALL_DONE",
    ]
    return "\n".join(lines)


def gen_extract(cfg, extract_dir):
    """拆解 ops 大包，拿到 run_package/ 下各组件 .run。"""
    po = cfg.pkg("good", cfg.ops_type)
    return (
        "#!/bin/bash\n"
        "set -x\n"
        f"{po} --noexec --extract={extract_dir} 2>&1\n"
        "echo EXTRACT_RC=$?\n"
        "echo EXTRACT_DONE"
    )


def gen_replace(cfg, inst, extract_dir, components):
    """
    用 good 的组件 .run 覆盖组合目录（--full）.
    components: 例如 ['hccl','hixl','cv'] 或 ['transformer'].
    """
    inst_path = cfg.install_root + "/" + inst
    lines = [
        "#!/bin/bash",
        "set -x",
        f"export PATH={cfg.python_path}:$PATH",
        f"COMBO={inst_path}",
        f"EXTRACT={extract_dir}",
        "source $COMBO/cann/set_env.sh",
        "source $COMBO/nnal/atb/set_env.sh",
    ]
    for c in components:
        run = f"{extract_dir}/run_package/cann-{cfg.ops_type}-{c}_{cfg.cann_version}_linux-aarch64.run"
        lines += [f"{run} --full --install-path=$COMBO --quiet 2>&1",
                  f"echo REPLACE_{c.upper()}_RC=$?"]
    lines += ["echo REPLACE_DONE"]
    return "\n".join(lines)


def gen_start(cfg, inst):
    """vLLM 启动脚本：source 环境 + nohup 启动 + 记录 PID 到 pidfile（供按 PID 清理）。"""
    first = cfg.start_command.splitlines()[0] if cfg.start_command else "vllm serve"
    return (
        "#!/bin/bash\n"
        f"export PATH={cfg.python_path}:$PATH\n"
        f"source {cfg.install_root}/{inst}/cann/set_env.sh\n"
        f"source {cfg.install_root}/{inst}/nnal/atb/set_env.sh\n"
        f"cd {cfg.vllm_dir}\n"
        f"mkdir -p {cfg.workdir}/run\n"
        f"nohup {first} --port {cfg.port_num} > {cfg.workdir}/logs/vllm_{inst}.log 2>&1 &\n"
        "VPID=$!\n"
        f'echo "$VPID" > {cfg.workdir}/run/{inst}.pid\n'
        f'echo "PID=$VPID (saved to {cfg.workdir}/run/{inst}.pid)"'
    )


def gen_cleanup(cfg):
    """按 PID 精确清理本 skill 启动的 vLLM 进程树（禁止全局 pkill，避免误杀他人进程）。"""
    return (
        "#!/bin/bash\n"
        "# 按 PID 精确清理，禁止 pkill -f 模糊匹配（可能误杀别人的卡/进程）\n"
        "kill_tree() {\n"
        "  local p=$1\n"
        "  local child\n"
        "  for child in $(pgrep -P \"$p\" 2>/dev/null); do\n"
        "    kill_tree \"$child\"\n"
        "  done\n"
        "  [ -n \"$p\" ] && kill -9 \"$p\" 2>/dev/null\n"
        "}\n"
        f"RUN_DIR={cfg.workdir}/run\n"
        "for pf in \"$RUN_DIR\"/*.pid; do\n"
        "  [ -e \"$pf\" ] || continue\n"
        "  pid=$(cat \"$pf\")\n"
        "  echo \"killing tree of PID=$pid (pidfile $pf)\"\n"
        "  kill_tree \"$pid\"\n"
        "  rm -f \"$pf\"\n"
        "done\n"
        f"export PATH={cfg.python_path}:$PATH\n"
        "source " + cfg.install_root + "/good/cann/set_env.sh\n"
        "sleep 6\n"
        "npu-smi info"
    )


def gen_subpkg_combo(cfg, kind_tk, kind_nnal, kind_ops, inst):
    """
    细定位 tk / nnal 的组合安装（整包子包装，不走组件 --full）：
      - bad_tk   : bad tk   + good nnal + good ops
      - bad_nnal : good tk  + bad nnal  + good ops
    独立全新安装，严禁 cp -a。用于 combo1 正常时判断问题在 tk 还是 nnal。
    """
    pk = cfg.pkg(kind_tk, "toolkit")
    pn = cfg.pkg(kind_nnal, "nnal")
    po = cfg.pkg(kind_ops, cfg.ops_type)
    inst_path = cfg.install_root + "/" + inst
    lines = [
        "#!/bin/bash",
        "set -x",
        f"export PATH={cfg.python_path}:$PATH",
        f"INST={inst_path}",
        "rm -rf $INST",
        "mkdir -p $INST",
        "",
        f"# 组合: tk={kind_tk} nnal={kind_nnal} ops={kind_ops} (独立全新安装, 严禁 cp -a)",
        f"{pk} --install --install-path=$INST --quiet 2>&1",
        "echo TK_RC=$?",
        "source $INST/cann/set_env.sh",
        f"{pn} --install --install-path=$INST --quiet 2>&1",
        "echo NNAL_RC=$?",
        "source $INST/nnal/atb/set_env.sh",
        f"{po} --install --install-path=$INST --quiet 2>&1",
        "echo OPS_RC=$?",
        "",
        "echo ALL_DONE",
    ]
    return "\n".join(lines)


def gen_report(cfg):
    return "\n".join([
        "{:=^64}".format(" CANN 拆包定位最终报告 "),
        "定位时间 : <时间>",
        "服务器   : " + cfg.host,
        "CANN版本 : " + cfg.cann_version + "        Ops类型 : " + cfg.ops_type,
        "启动命令 : " + (cfg.start_command.splitlines()[0] if cfg.start_command else ""),
        "",
        "[判定标准 (用户定义)]",
        "异常现象 : <乱码/具体报错/性能问题...>",
        "验证方法 : <如何复现与判断正常/异常>",
        "",
        "[基线]",
        "全 good 包 -> <正常>",
        "全 bad 包  -> <异常>",
        "=> 问题在 CANN 包，可复现",
        "",
        "[子包二分]",
        "good tk + good nnal + bad ops  -> <异常/正常>",
        "=> <问题在 ops 子包 或 tk/nnal(不可拆)>",
        "",
        "[组件二分] (仅 ops 可独立替换的 6 组件)",
        "第一轮(组A=hccl+hixl+cv 换good, 组B保持bad)  -> <异常/正常> -> 问题在 <组A/组B>",
        "第二轮(组内逐个换good) -> 定位到组件: <组件名>",
        "(若两轮均定位不到 => 不在6组件内)",
        "",
        "[结论] (按实际路径三选一)",
        "- 问题在 tk/nnal 子包 => 直接报告定位到子包, 不再下拆",
        "- 问题在 ops 6 组件内 => 坏组件: cann-<ops-type>-<组件名>, 用 good 该组件 --full 覆盖重启验证",
        "- 问题在 ops 但不在 6 组件内 => 建议整包替换 bad -> good",
    ])


def main():
    cfg = Config()
    cfg.prompt()

    print()
    print("=" * 64)
    print("配置确认")
    print("=" * 64)
    print(cfg.summary())

    ok = input("\n确认无误？(y/N): ").strip().lower()
    if ok != "y":
        print("已取消。")
        sys.exit(0)

    out_dir = os.path.join(os.getcwd(), "generated")
    os.makedirs(out_dir, exist_ok=True)

    extract_dir = cfg.install_root + "/extract"
    scripts = {
        # 0) 基线
        "install_good.sh": gen_full_install(cfg, "good", "good"),
        "install_bad.sh": gen_full_install(cfg, "bad", "bad"),
        # 1) 子包二分
        "install_combo1.sh": gen_combo_install(cfg, "combo1"),
        # 1b) tk/nnal 细定位 (combo1 正常时判断 tk 还是 nnal, 不可拆)
        "install_bad_tk.sh": gen_subpkg_combo(cfg, "bad", "good", "good", "bad_tk"),
        "install_bad_nnal.sh": gen_subpkg_combo(cfg, "good", "bad", "good", "bad_nnal"),
        # 3) 拆 ops
        "extract_ops.sh": gen_extract(cfg, extract_dir),
        # 3a) 第一轮: 组A(hccl+hixl+cv) 换 good
        "replace_groupA.sh": gen_replace(cfg, "combo1", extract_dir, ["hccl", "hixl", "cv"]),
        # 3b) 第二轮 A 组内逐个
        "replace_groupA_hccl.sh": gen_replace(cfg, "combo1", extract_dir, ["hccl"]),
        "replace_groupA_hixl.sh": gen_replace(cfg, "combo1", extract_dir, ["hixl"]),
        "replace_groupA_cv.sh": gen_replace(cfg, "combo1", extract_dir, ["cv"]),
        # 3c) 第二轮 B 组内逐个
        "replace_groupB_math.sh": gen_replace(cfg, "combo1", extract_dir, ["math"]),
        "replace_groupB_nn.sh": gen_replace(cfg, "combo1", extract_dir, ["nn"]),
        "replace_groupB_transformer.sh": gen_replace(cfg, "combo1", extract_dir, ["transformer"]),
        # 4) 启动 (各环境)
        "start_good.sh": gen_start(cfg, "good"),
        "start_bad.sh": gen_start(cfg, "bad"),
        "start_combo1.sh": gen_start(cfg, "combo1"),
        "start_bad_tk.sh": gen_start(cfg, "bad_tk"),
        "start_bad_nnal.sh": gen_start(cfg, "bad_nnal"),
        # 5) 按 PID 精确清理
        "cleanup.sh": gen_cleanup(cfg),
        # 报告
        "report.txt": gen_report(cfg),
    }

    for name, content in scripts.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write(content)

    print()
    print("=" * 64)
    print(f"已生成脚本到: {out_dir}")
    print("=" * 64)
    for name in scripts:
        print("  -", name)

    print()
    print("下一步：把 good / bad 两套 3 个 .run 上传到服务器，再把 generated/*.sh 上传到工作目录，")
    print("在服务器上 chmod +x 后，按 SKILL 工作流依次执行：")
    print()
    print("0) 基线")
    print("   bash install_good.sh ; 看日志 RC=0 + ALL_DONE -> bash start_good.sh ; " + cfg.verify_cmd + " (期望正常)")
    print("   bash cleanup.sh")
    print("   bash install_bad.sh  ;                     -> bash start_bad.sh  ; " + cfg.verify_cmd + " (期望异常)")
    print("   bash cleanup.sh")
    print()
    print("1) 子包二分")
    print("   bash install_combo1.sh ; bash start_combo1.sh ; " + cfg.verify_cmd)
    print("     - 异常 => 问题在 ops, 进入第 2 步")
    print("     - 正常 => 分别 bash install_bad_tk.sh / install_bad_nnal.sh 定位是 tk 还是 nnal, 然后直接生成报告(不再下拆)")
    print()
    print("2) ops 组件二分 (仅 6 个可独立替换组件)")
    print("   bash extract_ops.sh  (拿到 run_package/)")
    print("   第一轮: bash replace_groupA.sh ; bash start_combo1.sh ; " + cfg.verify_cmd)
    print("     - 异常 => 问题在组A(hccl/hixl/cv)")
    print("     - 正常 => 问题在组B(math/nn/transformer)")
    print("   第二轮 (组内逐个): 每换一个就 bash start_combo1.sh + " + cfg.verify_cmd + " 看是否恢复正常")
    print("     组A: replace_groupA_hccl.sh / replace_groupA_hixl.sh / replace_groupA_cv.sh")
    print("     组B: replace_groupB_math.sh / replace_groupB_nn.sh / replace_groupB_transformer.sh")
    print("   - 两轮都定位不到 => 不在 6 组件内, 直接生成报告建议整包换")


if __name__ == "__main__":
    main()
