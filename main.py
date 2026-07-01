"""Modular RAG MCP Server — entry point."""

import sys
from pathlib import Path

# 本项目采用 src-layout（业务包都在 src/ 下，见 pyproject.toml 的 packages.find）。
# `pip install -e .` 之后包已可导入；但为了支持在未安装环境下 `python main.py` 直接运行，
# 这里把 src/ 手动插到 sys.path 最前面，确保 `import core` 等能被解析。
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main() -> None:
    # 延迟到函数内部再 import：settings 依赖 A3 才落地的 core.settings 模块，
    # 放在顶层会让「仅导入 main」在早期阶段就失败。函数内导入把这个耦合限制在真正运行时。
    from core.settings import load_settings

    # 唯一真源是 config/settings.yaml；缺关键字段时 load_settings 会 fail-fast 抛可读错误（A3 实现）。
    settings = load_settings("config/settings.yaml")
    print(f"Modular RAG MCP Server starting — LLM provider: {settings.llm.provider}")


if __name__ == "__main__":
    main()
