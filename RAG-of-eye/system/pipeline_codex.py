"""
Pipeline Codex
===============

按照 README 的建议顺序，一键依次执行 ingest → mining → builder。
默认行为等价于依次运行：

    python system/ingest_pipeline.py
    python system/mining_task.py
    python system/myopia_builder.py

可以通过命令行参数选择性执行某些步骤。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_DIR = PROJECT_ROOT / "system"
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    script: Path
    description: str

    def command(self) -> List[str]:
        return [sys.executable, str(self.script)]


STEPS: List[PipelineStep] = [
    PipelineStep("ingest", SYSTEM_DIR / "ingest_pipeline.py", "解析 PDF 并构建 Chroma"),
    PipelineStep("mining", SYSTEM_DIR / "mining_task.py", "批量抽取 medical_config.json"),
    PipelineStep("builder", SYSTEM_DIR / "myopia_builder.py", "构建 myopia_db.json"),
]
STEP_LOOKUP = {step.name: step for step in STEPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按照固定顺序运行 Myopia Pipeline。")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=[step.name for step in STEPS],
        help="仅执行指定步骤，顺序按提供的列表执行（默认 ingest→mining→builder）。",
    )
    parser.add_argument(
        "--ignore-errors",
        action="store_true",
        help="若某个步骤失败，继续执行后续步骤（默认遇到错误立即停止）。",
    )
    return parser.parse_args()


def ensure_env_configured() -> None:
    if ENV_FILE.exists():
        print(f"[OK] 检测到环境文件: {ENV_FILE}")
    else:
        print(f"[WARN] 未找到 {ENV_FILE}，请确认 API KEY/Embedding 已配置。")


def iter_steps(selected: Iterable[str] | None) -> Iterable[PipelineStep]:
    if not selected:
        return STEPS
    return [STEP_LOOKUP[name] for name in selected]


def run_step(step: PipelineStep) -> int:
    print(f"\n=== ▶ {step.name.upper()} | {step.description} ===")
    print(f"[CMD] {' '.join(step.command())}")
    result = subprocess.run(step.command(), cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print(f"[DONE] {step.name} 完成。")
    else:
        print(f"[FAIL] {step.name} 失败，返回码 {result.returncode}")
    return result.returncode


def main() -> int:
    args = parse_args()
    ensure_env_configured()

    for step in iter_steps(args.steps):
        code = run_step(step)
        if code != 0 and not args.ignore_errors:
            return code

    print("\n✅ Pipeline Codex 全部任务执行完毕。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
