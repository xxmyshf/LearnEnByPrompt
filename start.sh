#!/usr/bin/env bash
#
# start.sh — 启动 analyzePrompt Web 服务（主服务 + 翻译服务 + 可选 TTS）
#
# 用法:
#   ./start.sh                 # 启动主服务（自动启动翻译服务）
#   ./start.sh --tts            # 同时启动 TTS 语音合成服务
#   ./start.sh --port 9000      # 自定义端口
#   ./start.sh --host 127.0.0.1 # 自定义监听地址
#   ./start.sh --force          # 先杀死占用端口的旧进程再启动
#   ./start.sh --tts --port 9000 --force
#   PORT=9000 ./start.sh        # 环境变量方式也有效
#
# 环境变量:
#   PORT             主服务端口 (默认 8030, 同 config.json)
#   HOST             监听地址 (默认 127.0.0.1)
#   TRANSLATE_PORT   翻译服务端口 (默认 8053)
#   TTS_PORT         TTS 服务端口 (默认 8052)
#   START_TTS=1      等同于 --tts
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
VENV_PY="$PROJECT_ROOT/.venv/bin/python"
TTS_VENV_PY="$PROJECT_ROOT/Matcha-TTS/.venv/bin/python"

# ---------- 参数解析 ----------
START_TTS=false
FORCE=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tts)
            START_TTS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --port)
            EXTRA_ARGS+=("--port" "$2")
            shift 2
            ;;
        --host)
            EXTRA_ARGS+=("--host" "$2")
            shift 2
            ;;
        --help|-h)
            head -22 "$0" | tail -18
            exit 0
            ;;
        *)
            error "未知参数: $1"
            echo "用法: ./start.sh [--tts] [--force] [--port PORT] [--host HOST]"
            exit 1
            ;;
    esac
done

# 环境变量也可以触发 TTS
if [[ "${START_TTS:-0}" == "1" ]]; then
    START_TTS=true
fi

# ---------- 前置检查 ----------
if ! command -v uv &>/dev/null; then
    error "未找到 uv，请先安装:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [[ ! -f "$VENV_PY" ]]; then
    error "主项目 venv 不存在: $VENV_PY"
    error "请先运行:  ./install_deps.sh"
    exit 1
fi

if [[ ! -f "$PROJECT_ROOT/server.py" ]]; then
    error "server.py 不存在于 $PROJECT_ROOT"
    exit 1
fi

# ---------- 计算目标端口 ----------
# 从 --port 参数或 PORT 环境变量获取端口号，用于冲突检测
TARGET_PORT="${PORT:-8030}"
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--port" ]] && [[ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
        TARGET_PORT="${EXTRA_ARGS[$((i+1))]}"
    fi
done

# ---------- 端口占用检测 ----------
# 返回占用端口的 PID（无占用则返回空）
port_pid() {
    local port="$1"
    local pid=""
    if command -v lsof &>/dev/null; then
        pid="$(lsof -t -i :"$port" 2>/dev/null | head -1 || true)"
    elif command -v fuser &>/dev/null; then
        pid="$(fuser "$port/tcp" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | head -1 || true)"
    fi
    echo "$pid"
}

# 通过尝试 bind 检测端口是否可用（不依赖 lsof 权限，更可靠）
# 返回 0 = 端口被占用, 返回 1 = 端口空闲
port_in_use() {
    local port="$1"
    if "$VENV_PY" -c "
import socket, sys, errno
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('127.0.0.1', $port))
    s.close()
    sys.exit(0)  # bind 成功 = 端口空闲
except OSError as e:
    try:
        s.close()
    except Exception:
        pass
    if e.errno in (errno.EACCES, errno.EPERM):
        sys.exit(0)  # 权限不足，无法判断，视为空闲
    sys.exit(1)     # EADDRINUSE = 端口被占用
" 2>/dev/null; then
        return 1  # 端口空闲
    else
        return 0  # 端口被占用
    fi
}

if port_in_use "$TARGET_PORT"; then
    OCCUPYING_PID="$(port_pid "$TARGET_PORT")"
    if $FORCE; then
        if [[ -n "$OCCUPYING_PID" ]]; then
            warn "端口 $TARGET_PORT 被占用 (PID $OCCUPYING_PID)，--force 模式，正在终止..."
            kill -TERM "$OCCUPYING_PID" 2>/dev/null || true
            sleep 1
            if kill -0 "$OCCUPYING_PID" 2>/dev/null; then
                warn "进程未退出，强制杀死 (PID $OCCUPYING_PID)..."
                kill -9 "$OCCUPYING_PID" 2>/dev/null || true
                sleep 1
            fi
            info "旧进程已终止。"
        else
            warn "端口 $TARGET_PORT 已被占用但无法确定进程 PID"
            warn "请手动排查后重试"
            exit 1
        fi
        # 确认端口已释放
        if port_in_use "$TARGET_PORT"; then
            error "端口 $TARGET_PORT 仍被占用，无法启动"
            exit 1
        fi
    else
        if [[ -n "$OCCUPYING_PID" ]]; then
            error "端口 $TARGET_PORT 已被占用 (PID $OCCUPYING_PID)"
        else
            error "端口 $TARGET_PORT 已被占用"
        fi
        error "请先停止占用该端口的服务，或使用 --force 自动终止:"
        error "  ./start.sh --force"
        error "或使用其他端口:"
        error "  ./start.sh --port 9000"
        exit 1
    fi
fi

# ---------- 子进程 PID 跟踪 ----------
TTS_PID=""
MAIN_PID=""
_CLEANUP_DONE=0

cleanup() {
    [[ $_CLEANUP_DONE -eq 1 ]] && return 0
    _CLEANUP_DONE=1
    echo ""
    info "正在停止服务..."
    if [[ -n "$TTS_PID" ]] && kill -0 "$TTS_PID" 2>/dev/null; then
        info "停止 TTS 服务 (PID $TTS_PID)..."
        kill -TERM "$TTS_PID" 2>/dev/null || true
        wait "$TTS_PID" 2>/dev/null || true
    fi
    if [[ -n "$MAIN_PID" ]] && kill -0 "$MAIN_PID" 2>/dev/null; then
        info "停止主服务 (PID $MAIN_PID)..."
        kill -TERM "$MAIN_PID" 2>/dev/null || true
        wait "$MAIN_PID" 2>/dev/null || true
    fi
    info "已停止。"
}
trap cleanup EXIT INT TERM

# ---------- 可选: 启动 TTS 服务 ----------
if $START_TTS; then
    if [[ ! -f "$PROJECT_ROOT/tts_server.py" ]]; then
        warn "tts_server.py 不存在，跳过 TTS 启动"
        START_TTS=false
    elif [[ ! -f "$TTS_VENV_PY" ]]; then
        warn "Matcha-TTS venv 不存在: $TTS_VENV_PY"
        warn "请先运行 ./install_deps.sh 安装 Matcha-TTS 依赖"
        warn "跳过 TTS 启动，继续启动主服务..."
        START_TTS=false
    fi
fi

if $START_TTS; then
    TTS_PORT_VAL="${TTS_PORT:-8052}"
    # 检查 TTS 端口是否被占用
    TTS_OCCUPY="$(port_pid "$TTS_PORT_VAL")"
    if [[ -n "$TTS_OCCUPY" ]]; then
        warn "TTS 端口 $TTS_PORT_VAL 被占用 (PID $TTS_OCCUPY)，跳过 TTS 启动"
        START_TTS=false
    fi
fi

if $START_TTS; then
    info "=== 启动 TTS 语音合成服务 (端口 $TTS_PORT_VAL) ==="
    export TTS_PORT="$TTS_PORT_VAL"
    export PYTHONPATH="$PROJECT_ROOT/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
    "$TTS_VENV_PY" "$PROJECT_ROOT/tts_server.py" &
    TTS_PID=$!
    info "TTS 服务已启动 (PID $TTS_PID)，等待模型加载..."
    # 等待 TTS 健康检查通过
    TTS_DEADLINE=$((SECONDS + 300))
    while [[ $SECONDS -lt $TTS_DEADLINE ]]; do
        if ! kill -0 "$TTS_PID" 2>/dev/null; then
            error "TTS 进程已退出"
            TTS_PID=""
            break
        fi
        if curl -sf "http://127.0.0.1:${TTS_PORT_VAL}/health" >/dev/null 2>&1; then
            info "TTS 服务已就绪"
            break
        fi
        sleep 2
    done
    [[ -n "$TTS_PID" ]] || warn "TTS 服务未就绪，继续启动主服务"
    # 恢复 PYTHONPATH，避免影响主服务
    unset PYTHONPATH
fi

# ---------- 启动主服务 ----------
info "=== 启动主服务 ==="
info "翻译服务将由主服务自动启动"

# 计算显示用的 host/port（从参数或环境变量取）
DISPLAY_HOST="${HOST:-127.0.0.1}"
DISPLAY_PORT="$TARGET_PORT"
for ((i=0; i<${#EXTRA_ARGS[@]}; i++)); do
    if [[ "${EXTRA_ARGS[$i]}" == "--host" ]] && [[ $((i+1)) -lt ${#EXTRA_ARGS[@]} ]]; then
        DISPLAY_HOST="${EXTRA_ARGS[$((i+1))]}"
    fi
done
info "浏览器访问: http://${DISPLAY_HOST}:${DISPLAY_PORT}"
info "按 Ctrl+C 停止所有服务"

# 在后台启动主服务，前台等待（这样 trap 能正确清理所有子进程）
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    "$VENV_PY" -u "$PROJECT_ROOT/server.py" "${EXTRA_ARGS[@]}" &
else
    "$VENV_PY" -u "$PROJECT_ROOT/server.py" &
fi
MAIN_PID=$!

# 等待主服务退出（Ctrl+C 或 kill 会触发 cleanup）
wait "$MAIN_PID"
EXIT_CODE=$?
exit $EXIT_CODE
