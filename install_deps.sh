#!/usr/bin/env bash
#
# install_deps.sh — 一键创建 venv 并安装主项目与 Matcha-TTS 子模块的全部依赖
#
# 用法:
#   ./install_deps.sh            # 安装/更新全部依赖
#   ./install_deps.sh --dev      # 额外安装开发依赖 (pytest, pre-commit 等)
#   ./install_deps.sh --check    # 仅验证安装，不安装
#
set -euo pipefail

# ---------- 颜色输出 ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------- 路径与变量 ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
SUBMODULE_DIR="$PROJECT_ROOT/Matcha-TTS"

UV="${UV:-uv}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export UV_CACHE_DIR

DEV_MODE=false
CHECK_ONLY=false
case "${1:-}" in
    --dev)   DEV_MODE=true ;;
    --check) CHECK_ONLY=true ;;
esac

# ---------- 前置检查 ----------
if ! command -v "$UV" &>/dev/null; then
    error "未找到 uv，请先安装:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ---------- 确保 Matcha-TTS 子模块存在 ----------
if [[ ! -d "$SUBMODULE_DIR" ]]; then
    if $CHECK_ONLY; then
        error "Matcha-TTS 子模块目录不存在，请先运行: git submodule update --init --recursive"
    else
        warn "Matcha-TTS 子模块目录不存在，尝试初始化..."
        git -C "$PROJECT_ROOT" submodule update --init --recursive
    fi
fi

# ============================================================
# 安装步骤（--check 模式跳过）
# ============================================================
if ! $CHECK_ONLY; then

# ---------- 主项目依赖安装 ----------
info "=== 安装主项目 (analyzePrompt) 依赖 ==="

if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
    info "创建主项目 venv (Python 3.10)..."
    cd "$PROJECT_ROOT"
    "$UV" venv --python 3.10 .venv
fi

cd "$PROJECT_ROOT"
"$UV" pip install --python .venv/bin/python \
    fastapi httpx "uvicorn[standard]" transformers sentencepiece protobuf tiktoken torch

if $DEV_MODE; then
    info "安装开发依赖..."
    "$UV" pip install --python .venv/bin/python pytest pre-commit ruff black
fi

info "主项目依赖安装完成。"

# ---------- Matcha-TTS 子模块依赖安装 ----------
info "=== 安装 Matcha-TTS 子模块依赖 ==="

if [[ ! -d "$SUBMODULE_DIR/.venv" ]]; then
    info "创建 Matcha-TTS venv (Python 3.10)..."
    "$UV" venv --python 3.10 "$SUBMODULE_DIR/.venv"
fi

cd "$SUBMODULE_DIR"
"$UV" pip install --python .venv/bin/python -e .

info "Matcha-TTS 依赖安装完成。"

fi  # end !CHECK_ONLY

# ============================================================
# 验证安装
# ============================================================
info "=== 验证依赖安装 ==="

verify_import() {
    local venv_python="$1"
    local module="$2"
    if "$venv_python" -c "import $module" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $module"
    else
        echo -e "  ${RED}✗${NC} $module"
    fi
}

echo "主项目:"
verify_import "$PROJECT_ROOT/.venv/bin/python" "fastapi"
verify_import "$PROJECT_ROOT/.venv/bin/python" "uvicorn"
verify_import "$PROJECT_ROOT/.venv/bin/python" "httpx"
verify_import "$PROJECT_ROOT/.venv/bin/python" "torch"
verify_import "$PROJECT_ROOT/.venv/bin/python" "transformers"
verify_import "$PROJECT_ROOT/.venv/bin/python" "sentencepiece"

echo "Matcha-TTS:"
verify_import "$SUBMODULE_DIR/.venv/bin/python" "numpy"
verify_import "$SUBMODULE_DIR/.venv/bin/python" "torch"
verify_import "$SUBMODULE_DIR/.venv/bin/python" "soundfile"
verify_import "$SUBMODULE_DIR/.venv/bin/python" "matcha"
verify_import "$SUBMODULE_DIR/.venv/bin/python" "fastapi"

# ============================================================
# 已知问题检查
# ============================================================
info "=== 检查已知兼容性问题 ==="

#
# 问题 1: PyTorch 2.6+ 将 torch.load 的 weights_only 默认值改为 True，
#          导致 Matcha-TTS 的旧格式 checkpoint 无法加载。
#          tts_server.py 已内置 monkeypatch 修复此问题。
#
TTS_SERVER="$PROJECT_ROOT/tts_server.py"
if [[ -f "$TTS_SERVER" ]]; then
    if grep -q "_patched_torch_load" "$TTS_SERVER"; then
        echo -e "  ${GREEN}✓${NC} torch.load weights_only patch (PyTorch 2.6+ compat)"
    else
        warn "tts_server.py 缺少 torch.load weights_only patch"
        warn "  Matcha-TTS checkpoint 在 PyTorch 2.6+ 下无法加载"
        warn "  请确保 tts_server.py 包含 weights_only=False 的 monkeypatch"
    fi

    #
    # 问题 2: Matcha-TTS 的 synthesise() 使用 @torch.inference_mode() 装饰，
    #          生成的 mel tensor 处于 inference mode，传给 vocoder 时会触发:
    #          "Inference tensors cannot be saved for backward"
    #          tts_server.py 已用 torch.no_grad() + mel.clone().detach() 修复。
    #
    if grep -q "clone().detach()" "$TTS_SERVER"; then
        echo -e "  ${GREEN}✓${NC} inference_mode tensor detach (vocoder compat)"
    else
        warn "tts_server.py 缺少 inference_mode tensor detach 修复"
        warn "  synthesize 端点会返回 500 Internal Server Error"
        warn "  需要在 synthesize 中用 torch.no_grad() 包裹并对 mel 做 clone().detach()"
    fi
else
    warn "tts_server.py 不存在于 $TTS_SERVER"
fi

info "全部完成！"
