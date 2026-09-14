#!/usr/bin/env python3
"""
collect_prompts.py -- 收集本机 AI 编程助手 (Codex、Claude Code 等) 的用户输入提示词,
按 天 / 周 / 月 分组, 生成纯文本报告。

用法:
  python3 collect_prompts.py                  # 默认: 今天
  python3 collect_prompts.py --range day       # 仅今天
  python3 collect_prompts.py --range week      # 本周 (周一到周日)
  python3 collect_prompts.py --range month     # 本月 (自然月)
  python3 collect_prompts.py --range all       # 磁盘上的全部记录
  python3 collect_prompts.py --range 2026-09  # 指定月份
  python3 collect_prompts.py --range 2026-09-10  # 指定日期
  python3 collect_prompts.py --range 2026-W36  # 指定 ISO 周
  python3 collect_prompts.py -o /path/to/report.txt  # 自定义输出路径
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable

HOME = Path.home()

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------

@dataclass(frozen=True)  # frozen=True 使实例不可变, 可作为 set 元素 / 字典 key
class Prompt:
    """单条用户提示词记录。

    Attributes:
        agent:  来源 Agent 名称, 如 "Codex"、"Claude Code"
        ts:     提示词的时间戳 (UTC 带时区), 用于排序和范围过滤
        text:   用户实际输入的文本 (经过清洗和截断)
        session: 所属会话 ID (截断展示用)
        cwd:    该会话的工作目录 (如果能从日志中获取到)
    """
    agent: str
    ts: datetime
    text: str
    session: str
    cwd: str = ""


@dataclass
class AgentCollector:
    """所有 Agent 收集器的基类; 子类实现 collect() 读取各自文件。"""
    name: str

    def collect(self) -> list[Prompt]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 通用辅助函数
# ---------------------------------------------------------------------------

# 这些 XML 标签开头的内容是系统自动注入的上下文 (非用户真实输入),
# 收集时应跳过, 避免把环境信息当成用户提示词。
_SYSTEM_TAGS = (
    "<environment_context>",
    "<collaboration_mode>",
    "<skills_instructions>",
    "<permissions_instructions>",
    "<user_instructions>",
    "<developer_instructions>",
    "<goal_context>",
)


def _is_system(text: str) -> bool:
    """判断文本是否以系统注入标签开头, 是则返回 True (需跳过)。"""
    stripped = text.lstrip()
    return any(stripped.startswith(tag) for tag in _SYSTEM_TAGS)


def _clean(text: str) -> str:
    """清洗提示词文本: 去首尾空白, 超过 3000 字符则截断, 换行缩进 4 格。"""
    text = text.strip()
    if len(text) > 3000:
        text = text[:3000] + " [...truncated]"
    return text.replace("\n", "\n    ")


def _ts_from_unix(seconds: float, /, ms: bool = False) -> datetime:
    """将 Unix 时间戳转为 UTC 带时区的 datetime。

    Args:
        seconds: Unix 时间戳
        ms: True 表示入参是毫秒, 会自动除以 1000 转为秒
    Returns:
        带时区的 datetime; 若转换异常则返回 epoch (1970-01-01)
    """
    if ms:
        seconds = seconds / 1000.0
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


def _ts_from_iso(s: str) -> datetime:
    """将 ISO 8601 字符串 (如 '2026-07-22T04:51:52.716Z') 转为 datetime。

    Python 3.11+ 的 fromisoformat 不认 'Z' 后缀, 这里先把 'Z' 替换为 '+00:00'。
    解析失败时返回 epoch。
    """
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return datetime.fromtimestamp(0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Codex 收集器 -- 读取 ~/.codex/ 下的历史和会话文件
# ---------------------------------------------------------------------------

class CodexCollector(AgentCollector):
    """读取 ~/.codex/history.jsonl 和 ~/.codex/sessions/**/*.jsonl。

    Codex 的数据分两层:
      1. history.jsonl  -- 轻量提示词日志, 每行一条用户输入
         格式: {"session_id":"...", "ts":1784695578, "text":"用户输入"}
         ts 是 Unix 秒级时间戳。
      2. sessions/      -- 完整会话 rollout 文件, 含系统指令、工具调用等全部细节
         用户消息在 type=response_item 且 payload.role=="user" 的条目中,
         payload.content 是一个列表, 其中 type=="input_text" 的 text 字段是用户输入。
         session_meta 条目提供 session_id 和 cwd。
    """

    def __init__(self):
        super().__init__(name="Codex")
        self.codex_dir = HOME / ".codex"

    def collect(self) -> list[Prompt]:
        """合并 history.jsonl 和 session 文件的结果并去重。

        去重 key 为 (session_id, text[:200]) -- 因为同一条提示词会同时出现在
        history.jsonl (秒级 ts) 和 session 文件 (毫秒级 ts) 中,
        时间戳可能有微小差异, 所以只用 session+text 做去重。
        """
        prompts: list[Prompt] = []
        prompts.extend(self._from_history())
        prompts.extend(self._from_sessions())
        seen: set = set()
        unique: list[Prompt] = []
        for p in prompts:
            key = (p.session, p.text[:200])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def _from_history(self) -> list[Prompt]:
        """从 ~/.codex/history.jsonl 提取用户提示词 (快速路径, 无 cwd 信息)。"""
        path = self.codex_dir / "history.jsonl"
        if not path.exists():
            return []
        out: list[Prompt] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过无法解析的行
            text = obj.get("text", "")
            if not text or _is_system(text):
                continue  # 跳过空文本和系统注入内容
            ts = _ts_from_unix(obj.get("ts", 0))
            out.append(Prompt(self.name, ts, _clean(text), obj.get("session_id", "")))
        return out

    def _from_sessions(self) -> list[Prompt]:
        """从 ~/.codex/sessions/**/*.jsonl rollout 文件提取用户提示词 (含 cwd)。"""
        sessions_dir = self.codex_dir / "sessions"
        if not sessions_dir.exists():
            return []
        out: list[Prompt] = []
        for fpath in sorted(sessions_dir.rglob("*.jsonl")):
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # 每个 rollout 文件开头有 session_meta, 记录 session_id 和 cwd
            session_id = ""
            cwd = ""
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                typ = obj.get("type", "")
                # 会话元数据行: 提取 session_id 和工作目录
                if typ == "session_meta":
                    p = obj.get("payload", {})
                    session_id = p.get("session_id", p.get("id", ""))
                    cwd = p.get("cwd", "")
                # 响应条目行: 从中筛选 role=="user" 的消息
                if typ == "response_item":
                    payload = obj.get("payload", {})
                    if payload.get("type") == "message" and payload.get("role") == "user":
                        content_list = payload.get("content", [])
                        for item in content_list:
                            if isinstance(item, dict) and item.get("type") == "input_text":
                                raw = item.get("text", "")
                                if raw and not _is_system(raw):
                                    ts = _ts_from_iso(obj.get("timestamp", ""))
                                    out.append(Prompt(
                                        self.name, ts, _clean(raw),
                                        session_id or obj.get("session_id", ""),
                                        cwd,
                                    ))
        return out


# ---------------------------------------------------------------------------
# Claude Code 收集器 -- 读取 ~/.claude/ 下的历史和项目会话文件
# ---------------------------------------------------------------------------

class ClaudeCodeCollector(AgentCollector):
    """读取 ~/.claude/history.jsonl 和 ~/.claude/projects/**/*.jsonl。

    Claude Code 的数据也分两层:
      1. history.jsonl  -- 轻量提示词日志
         格式: {"display":"用户输入", "timestamp":1784693048201, "project":"/path", "sessionId":"..."}
         timestamp 是毫秒级。
      2. projects/<project>/*.jsonl -- 完整会话文件
         用户消息: type=="user" 且 message.role=="user"
         通过 origin.kind=="human" 区分真人输入 vs 工具返回结果
         message.content 可能是 str (纯文本) 或 list (含 tool_result 等, 需过滤)
    """

    def __init__(self):
        super().__init__(name="Claude Code")
        self.claude_dir = HOME / ".claude"

    def collect(self) -> list[Prompt]:
        """合并两个来源并去重, 逻辑同 CodexCollector。"""
        prompts: list[Prompt] = []
        prompts.extend(self._from_history())
        prompts.extend(self._from_projects())
        seen: set = set()
        unique: list[Prompt] = []
        for p in prompts:
            key = (p.session, p.text[:200])
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    def _from_history(self) -> list[Prompt]:
        """从 ~/.claude/history.jsonl 提取用户提示词。"""
        path = self.claude_dir / "history.jsonl"
        if not path.exists():
            return []
        out: list[Prompt] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("display", "")
            if not text or _is_system(text):
                continue
            ts_ms = obj.get("timestamp", 0)  # 毫秒级时间戳
            ts = _ts_from_unix(ts_ms, ms=True)
            out.append(Prompt(
                self.name, ts, _clean(text),
                obj.get("sessionId", ""),
                obj.get("project", ""),
            ))
        return out

    def _from_projects(self) -> list[Prompt]:
        """从 ~/.claude/projects/**/*.jsonl 会话文件提取真人用户提示词。"""
        projects_dir = self.claude_dir / "projects"
        if not projects_dir.exists():
            return []
        out: list[Prompt] = []
        for fpath in sorted(projects_dir.rglob("*.jsonl")):
            try:
                content = fpath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # 只看 type=="user" 的条目
                if obj.get("type") != "user":
                    continue
                # origin.kind=="human" 表示真人输入; 工具结果会被标记为其他 origin
                origin = obj.get("origin", {})
                if origin and origin.get("kind") != "human":
                    continue
                msg = obj.get("message", {})
                if msg.get("role") != "user":
                    continue
                # content 可能是 str (纯用户文本) 或 list (含 tool_result 等)
                raw = msg.get("content")
                if isinstance(raw, str):
                    text = raw
                elif isinstance(raw, list):
                    # 从列表中提取 type=="text" 的文本块, 跳过 tool_result
                    parts = []
                    for item in raw:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                    text = "\n".join(parts)
                else:
                    continue
                if not text or _is_system(text):
                    continue
                ts = _ts_from_iso(obj.get("timestamp", ""))
                out.append(Prompt(
                    self.name, ts, _clean(text),
                    obj.get("sessionId", ""),
                    obj.get("cwd", ""),
                ))
        return out


# ---------------------------------------------------------------------------
# Antigravity 收集器 -- 读取 ~/.gemini/antigravity/ 下的会话数据库
# ---------------------------------------------------------------------------

# Antigravity 的 step_type=14 对应 CORTEX_STEP_TYPE_USER_INPUT。
# 会话数据以 SQLite 数据库形式存储, 步骤内容为 protobuf 序列化的 BLOB。
# 用户输入文本位于 step_payload 顶层的 field 19 > field 2 中。
_AG_USER_INPUT_STEP = 14


def _pb_read_varint(data: bytes, i: int) -> tuple[int, int]:
    """读取 protobuf varint, 返回 (值, 下一个字节位置)。"""
    result = 0
    shift = 0
    while i < len(data):
        b = data[i]
        result |= (b & 0x7F) << shift
        i += 1
        if not (b & 0x80):
            break
        shift += 7
    return result, i


def _pb_parse_fields(data: bytes) -> dict[int, list[tuple[int, Any]]]:
    """解析 protobuf wire format, 返回 {字段号: [(wire_type, 值), ...]}。

    仅处理 varint (0) 和 length-delimited (2) 两种 wire type;
    32-bit (5) 和 64-bit (1) 跳过定长字节。解析失败时返回已解析部分。
    """
    fields: dict[int, list[tuple[int, Any]]] = {}
    i = 0
    while i < len(data):
        try:
            tag, i = _pb_read_varint(data, i)
            fn = tag >> 3
            wt = tag & 0x07
            if wt == 0:
                val, i = _pb_read_varint(data, i)
            elif wt == 2:
                length, i = _pb_read_varint(data, i)
                if i + length > len(data):
                    break
                val = data[i:i + length]
                i += length
            elif wt == 5:
                val = data[i:i + 4]
                i += 4
            elif wt == 1:
                val = data[i:i + 8]
                i += 8
            else:
                break
            fields.setdefault(fn, []).append((wt, val))
        except (IndexError, ValueError):
            break
    return fields


def _pb_get_bytes(fields: dict[int, list], fn: int) -> bytes | None:
    """取指定字段号的第一个 length-delimited 值。"""
    if fn in fields:
        for wt, v in fields[fn]:
            if wt == 2:
                return v  # type: ignore[return-value]
    return None


def _pb_get_varint(fields: dict[int, list], fn: int) -> int:
    """取指定字段号的第一个 varint 值, 找不到返回 0。"""
    if fn in fields:
        for wt, v in fields[fn]:
            if wt == 0:
                return v  # type: ignore[return-value]
    return 0


class AntigravityCollector(AgentCollector):
    """读取 ~/.gemini/antigravity/conversations/*.db。

    Google Antigravity (IDE + CLI `agy`) 将会话以 SQLite 数据库形式存储,
    每个会话一个 .db 文件, 文件名为会话 UUID。

    steps 表中 step_type=14 (CORTEX_STEP_TYPE_USER_INPUT) 的条目包含用户输入;
    step_payload 列是 protobuf 序列化的 BLOB, 结构为:
      - 顶层 field 1 (varint): step_type
      - 顶层 field 5 (bytes): 元数据, 内含时间戳和会话 UUID
          - field 1 (bytes): 时间戳对
              - field 1 (varint): Unix 秒级时间戳
          - field 20 (bytes): 会话上下文
              - field 4 (bytes): 会话 UUID
      - 顶层 field 19 (bytes): 用户输入载荷
          - field 2 (bytes): 用户输入文本 (UTF-8)
          - field 3 (bytes): 用户输入文本的副本 (嵌套在 field 1 中)

    工作目录从 conversation_summaries.db 的 workspace_uris 字段获取,
    或从 conversation_metadata.json 缓存文件获取。
    """

    def __init__(self):
        super().__init__(name="Antigravity")
        self.ag_dir = HOME / ".gemini" / "antigravity"
        self.convs_dir = self.ag_dir / "conversations"
        self.cli_dir = HOME / ".gemini" / "antigravity-cli"

    def collect(self) -> list[Prompt]:
        """扫描所有会话数据库, 提取 step_type=14 中的用户输入文本。"""
        if not self.convs_dir.exists():
            return []

        # 预加载工作目录映射 {conversation_id: cwd}
        cwd_map = self._load_cwd_map()

        out: list[Prompt] = []
        for db_path in sorted(self.convs_dir.glob("*.db")):
            conv_id = db_path.stem
            try:
                prompts = self._from_db(db_path, conv_id, cwd_map)
                out.extend(prompts)
            except Exception:
                continue  # 单个数据库损坏不影响其他
        return out

    def _load_cwd_map(self) -> dict[str, str]:
        """从 conversation_summaries.db 或 conversation_metadata.json
        构建 {conversation_id: cwd_path} 映射。"""
        cwd_map: dict[str, str] = {}

        # 优先从 conversation_summaries.db 读取
        summaries_db = self.cli_dir / "conversation_summaries.db"
        if summaries_db.exists():
            try:
                conn = sqlite3.connect(
                    f"file:{summaries_db}?immutable=1", uri=True)
                cur = conn.cursor()
                cur.execute(
                    "SELECT conversation_id, workspace_uris "
                    "FROM conversation_summaries")
                for cid, ws in cur.fetchall():
                    cwd_map[cid] = self._uri_to_path(ws)
                conn.close()
            except Exception:
                pass

        # 补充: 从 conversation_metadata.json 读取
        meta_path = self.cli_dir / "cache" / "conversation_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(
                    meta_path.read_text(encoding="utf-8", errors="replace"))
                for cid, info in meta.get("conversations", {}).items():
                    if cid not in cwd_map:
                        s = info.get("summary", {})
                        uris = s.get("WorkspaceURIs", [])
                        if uris:
                            cwd_map[cid] = self._uri_to_path(uris[0])
            except Exception:
                pass

        return cwd_map

    @staticmethod
    def _uri_to_path(uris: Any) -> str:
        """将 workspace_uris (JSON 字符串或列表) 转为路径字符串。"""
        if not uris:
            return ""
        if isinstance(uris, str):
            try:
                uris = json.loads(uris)
            except (json.JSONDecodeError, ValueError):
                return ""
        if isinstance(uris, list) and uris:
            uri = uris[0]
        elif isinstance(uris, str):
            uri = uris
        else:
            return ""
        if uri.startswith("file://"):
            uri = uri[7:]
        return uri

    def _from_db(self, db_path: Path, conv_id: str,
                 cwd_map: dict[str, str]) -> list[Prompt]:
        """从单个会话数据库提取用户输入提示词。"""
        # 工作目录: 优先用 cwd_map, 找不到则从 trajectory_metadata_blob 读取
        cwd = cwd_map.get(conv_id, "")
        if not cwd:
            cwd = self._cwd_from_trajectory(db_path)

        conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT step_payload FROM steps "
                "WHERE step_type=? ORDER BY idx",
                (_AG_USER_INPUT_STEP,))
            out: list[Prompt] = []
            for (payload,) in cur.fetchall():
                if not payload:
                    continue
                result = self._parse_payload(payload, conv_id, cwd)
                if result:
                    out.append(result)
            return out
        finally:
            conn.close()

    @staticmethod
    def _cwd_from_trajectory(db_path: Path) -> str:
        """从 trajectory_metadata_blob 表的 field 7 提取工作目录 URI。

        仅在 cwd_map 中找不到对应会话时作为回退使用。
        """
        try:
            conn = sqlite3.connect(f"file:{db_path}?immutable=1", uri=True)
            try:
                cur = conn.cursor()
                cur.execute(
                    'SELECT data FROM trajectory_metadata_blob '
                    "WHERE id='main' LIMIT 1")
                row = cur.fetchone()
                if row and row[0]:
                    fields = _pb_parse_fields(row[0])
                    uri = _pb_get_bytes(fields, 7)
                    if uri:
                        s = uri.decode("utf-8", errors="replace")
                        if s.startswith("file://"):
                            s = s[7:]
                        return s
            finally:
                conn.close()
        except Exception:
            pass
        return ""

    @staticmethod
    def _parse_payload(payload: bytes, conv_id: str,
                       cwd: str) -> Prompt | None:
        """解析 step_payload BLOB, 提取用户文本和时间戳。

        protobuf 结构:
          field 5 (bytes) > field 1 (bytes) > field 1 (varint) = Unix 秒
          field 19 (bytes) > field 2 (bytes) = 用户文本
        """
        top = _pb_parse_fields(payload)

        # 用户输入载荷在顶层 field 19
        user_payload = _pb_get_bytes(top, 19)
        if not user_payload:
            return None  # 无用户文本 (系统生成的步骤)

        user_fields = _pb_parse_fields(user_payload)
        user_text = _pb_get_bytes(user_fields, 2)

        # field 2 找不到时, 尝试 field 3 > field 1 (嵌套副本)
        if not user_text:
            f3 = _pb_get_bytes(user_fields, 3)
            if f3:
                f3_fields = _pb_parse_fields(f3)
                user_text = _pb_get_bytes(f3_fields, 1)

        if not user_text:
            return None

        try:
            text = user_text.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            return None
        if not text.strip() or _is_system(text):
            return None

        # 时间戳: 顶层 field 5 > field 1 > field 1 (Unix 秒)
        main = _pb_get_bytes(top, 5)
        ts_seconds = 0
        if main:
            main_fields = _pb_parse_fields(main)
            ts_data = _pb_get_bytes(main_fields, 1)
            if ts_data:
                ts_fields = _pb_parse_fields(ts_data)
                ts_seconds = _pb_get_varint(ts_fields, 1)

        ts = _ts_from_unix(ts_seconds)

        return Prompt("Antigravity", ts, _clean(text), conv_id, cwd)


# ---------------------------------------------------------------------------
# 时间范围解析
# ---------------------------------------------------------------------------

def _parse_range(rng: str) -> tuple[datetime, datetime, str]:
    """将 --range 参数解析为 (起始时间, 结束时间(不含), 标签)。

    支持的格式:
      "day"/"today"/""  -- 今天 00:00 到明天 00:00
      "week"            -- 本周一到下周一 (ISO 周)
      "month"           -- 本月 1 号到下月 1 号
      "all"             -- 1970 到现在
      "YYYY-MM"         -- 指定月份
      "YYYY-MM-DD"      -- 指定日期
      "YYYY-W##"        -- 指定 ISO 周

    返回的起始/结束时间都带本地时区, 以便和 UTC 时间戳正确比较。
    """
    now = datetime.now(timezone.utc)
    local_now = datetime.now()
    # 获取本地时区, 用于给 today_start 加时区信息 (prompt.ts 是 UTC 带时区的)
    tz = local_now.astimezone().tzinfo
    today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=tz)
    today_end = today_start + timedelta(days=1)

    if rng in ("day", "today", ""):
        return today_start, today_end, today_start.strftime("%Y-%m-%d")

    if rng == "week":
        # weekday() 返回 0=周一, 回退到本周一
        monday = today_start - timedelta(days=today_start.weekday())
        sunday_end = monday + timedelta(days=7)
        iso_week = monday.strftime("%G-W%V")
        return monday, sunday_end, iso_week

    if rng == "month":
        m_start = today_start.replace(day=1)
        if m_start.month == 12:
            m_end = m_start.replace(year=m_start.year + 1, month=1)
        else:
            m_end = m_start.replace(month=m_start.month + 1)
        return m_start, m_end, m_start.strftime("%Y-%m")

    if rng == "all":
        return datetime(1970, 1, 1, tzinfo=timezone.utc), now, "all-time"

    # 精确月份: YYYY-MM
    m = re.fullmatch(r"(\d{4})-(\d{2})", rng)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        start = datetime(y, mo, 1)
        if mo == 12:
            end = datetime(y + 1, 1, 1)
        else:
            end = datetime(y, mo + 1, 1)
        return start, end, rng

    # 精确日期: YYYY-MM-DD
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", rng)
    if m:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return d, d + timedelta(days=1), rng

    # ISO 周: YYYY-W##
    m = re.fullmatch(r"(\d{4})-W(\d{2})", rng)
    if m:
        y, w = int(m.group(1)), int(m.group(2))
        monday = datetime.fromisocalendar(y, w, 1)  # 返回该周一
        return monday, monday + timedelta(days=7), rng

    sys.exit(f"Unrecognised --range value: {rng}")


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def _fmt_ts(ts: datetime) -> str:
    """将 UTC 时间戳格式化为本地时间的可读字符串。"""
    local = ts.astimezone()
    return local.strftime("%Y-%m-%d %H:%M:%S")


def _local_date(ts: datetime) -> str:
    """将 UTC 时间戳转为本地日期字符串 (YYYY-MM-DD), 用于按天分组。"""
    return ts.astimezone().strftime("%Y-%m-%d")


def generate_report(prompts: list[Prompt], start: datetime, end: datetime, label: str) -> str:
    """生成纯文本报告字符串。

    报告结构:
      1. 头部 -- 时间范围、生成时间、总数、Agent 列表
      2. 按 Agent 统计 -- 每个 Agent 的提示词数量
      3. 按天统计 -- 每天的提示词数量
      4. 详细列表 -- 按天分组, 每天内部按时间排序, 逐条列出
    """
    lines: list[str] = []

    # --- 头部 ---
    lines.append("=" * 72)
    lines.append("AI Agent Prompt Collection Report")
    lines.append("=" * 72)
    lines.append(f"Range label  : {label}")
    lines.append(f"Period (UTC) : {start.strftime('%Y-%m-%d %H:%M')} -> "
                  f"{end.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Generated at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total prompts: {len(prompts)}")
    if prompts:
        agents = sorted({p.agent for p in prompts})
        lines.append(f"Agents found : {', '.join(agents)}")
    lines.append("")

    # 空结果直接返回
    if not prompts:
        lines.append("(no prompts found in this period)")
        return "\n".join(lines)

    # --- 按 Agent 统计 ---
    lines.append("-" * 72)
    lines.append("Summary by Agent")
    lines.append("-" * 72)
    by_agent: dict[str, int] = defaultdict(int)
    for p in prompts:
        by_agent[p.agent] += 1
    for agent in sorted(by_agent):
        lines.append(f"  {agent:20s}  {by_agent[agent]:>5d} prompts")
    lines.append("")

    # --- 按天统计 ---
    lines.append("-" * 72)
    lines.append("Summary by Day")
    lines.append("-" * 72)
    by_day: dict[str, int] = defaultdict(int)
    for p in prompts:
        by_day[_local_date(p.ts)] += 1
    for day in sorted(by_day):
        lines.append(f"  {day}   {by_day[day]:>5d} prompts")
    lines.append("")

    # --- 详细列表: 按天分组, 组内按时间排序 ---
    lines.append("-" * 72)
    lines.append("Detailed Prompts (grouped by day, chronological)")
    lines.append("-" * 72)

    day_groups: dict[str, list[Prompt]] = defaultdict(list)
    for p in prompts:
        day_groups[_local_date(p.ts)].append(p)

    for day in sorted(day_groups):
        day_prompts = sorted(day_groups[day], key=lambda p: p.ts)
        lines.append("")
        lines.append(f"### {day}  ({len(day_prompts)} prompts) {'#' * max(0, 60 - len(day))}")
        for i, p in enumerate(day_prompts, 1):
            lines.append("")
            lines.append(f"  [{i:03d}] {_fmt_ts(p.ts)}  [{p.agent}]  session={p.session[:12]}")
            if p.cwd:
                lines.append(f"       cwd: {p.cwd}")
            lines.append(f"       prompt:")
            lines.append(f"         {p.text}")
    lines.append("")
    lines.append("=" * 72)
    lines.append("End of report")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> int:
    """解析命令行参数, 收集所有 Agent 的提示词, 过滤并生成报告文件。"""
    parser = argparse.ArgumentParser(
        description="Collect AI agent user prompts by day/week/month.")
    parser.add_argument(
        "--range", "-r", default="day",
        help="Time range: day, week, month, all, YYYY-MM, YYYY-MM-DD, or YYYY-W## (default: day)")
    parser.add_argument(
        "--output", "-o", default="",
        help="Output file path (default: ~/prompt_report_<label>.txt)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    # 解析时间范围 -> (起始, 结束(不含), 标签)
    start, end, label = _parse_range(args.range)

    # 注册所有收集器; 未来新增 Agent 只需在这里追加一个子类实例
    collectors = [
        CodexCollector(),
        ClaudeCodeCollector(),
        AntigravityCollector(),
    ]

    # 逐个收集; 某个 Agent 失败不影响其他
    all_prompts: list[Prompt] = []
    for c in collectors:
        try:
            collected = c.collect()
            all_prompts.extend(collected)
        except Exception as exc:
            print(f"[warn] {c.name} collection failed: {exc}", file=sys.stderr)

    # 按时间范围过滤并排序
    filtered = [p for p in all_prompts if start <= p.ts < end]
    filtered.sort(key=lambda p: p.ts)

    # 生成报告文本
    report = generate_report(filtered, start, end, label)

    # 确定输出路径: 有 -o 用指定的, 否则用 ~/prompt_report_<label>.txt
    if args.output:
        out_path = Path(args.output)
    else:
        safe_label = label.replace("/", "_")
        out_path = HOME / f"prompt_report_{safe_label}.txt"

    out_path.write_text(report, encoding="utf-8")
    print(f"Collected {len(filtered)} prompts from {len(collectors)} agent(s).")
    print(f"Report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
