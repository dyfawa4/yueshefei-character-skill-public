from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


CATALOG_NAME = "resource_catalog.json"
MAX_CHUNK_CHARS = 2200
IGNORED_NAME_PARTS = (".bak", ".pre-")


@dataclass(frozen=True, slots=True)
class SkillChunk:
    resource_id: str
    path: str
    evidence_type: str
    authoritative: bool
    line_start: int
    line_end: int
    title: str
    text: str


@dataclass(frozen=True, slots=True)
class SkillResource:
    resource_id: str
    path: str
    scope: str
    evidence_type: str
    authoritative: bool
    crosscheck: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class SkillIndex:
    root: Path
    signature: tuple[tuple[str, int, int], ...]
    resources: dict[str, SkillResource]
    chunks: dict[str, tuple[SkillChunk, ...]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_formal_content(path: Path) -> bool:
    name = path.name
    return (
        path.is_file()
        and name != CATALOG_NAME
        and not any(part in name for part in IGNORED_NAME_PARTS)
    )


def skill_signature(root: Path) -> tuple[tuple[str, int, int], ...]:
    root = root.resolve()
    paths = [root / CATALOG_NAME]
    paths.extend(sorted(p for p in root.rglob("*") if _is_formal_content(p)))
    signature = []
    for path in paths:
        stat = path.stat()
        signature.append(
            (path.relative_to(root).as_posix(), stat.st_mtime_ns, stat.st_size)
        )
    return tuple(signature)


def _split_markdown(
    resource: SkillResource, text: str
) -> tuple[SkillChunk, ...]:
    lines = text.splitlines()
    chunks: list[SkillChunk] = []
    title = ""
    buffer: list[tuple[int, str]] = []
    table_header = ""

    def flush(*, header: str = "") -> None:
        nonlocal buffer
        while buffer and not buffer[0][1].strip():
            buffer.pop(0)
        while buffer and not buffer[-1][1].strip():
            buffer.pop()
        if not buffer:
            return
        body = "\n".join(line for _, line in buffer).strip()
        prefix = f"## {title}\n" if title else ""
        if header:
            prefix += header + "\n"
        chunk_text = (prefix + body).strip()
        chunks.append(
            SkillChunk(
                resource_id=resource.resource_id,
                path=resource.path,
                evidence_type=resource.evidence_type,
                authoritative=resource.authoritative,
                line_start=buffer[0][0],
                line_end=buffer[-1][0],
                title=title,
                text=chunk_text,
            )
        )
        buffer = []

    for line_no, line in enumerate(lines, 1):
        heading = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if heading:
            flush()
            title = heading.group(1).strip()
            table_header = ""
            continue
        if line.startswith("|") and line.rstrip().endswith("|"):
            flush()
            if re.match(r"^\|(?:\s*:?-+:?\s*\|)+$", line):
                continue
            if not table_header:
                table_header = line
                continue
            buffer.append((line_no, line))
            flush(header=table_header)
            continue
        table_header = ""
        is_list_item = bool(re.match(r"^\s*(?:[-*+] |\d+[.)]\s+)", line))
        if is_list_item and buffer:
            flush()
        if not line.strip():
            flush()
            continue
        projected = sum(len(item[1]) + 1 for item in buffer) + len(line)
        if buffer and projected > MAX_CHUNK_CHARS:
            flush()
        buffer.append((line_no, line))
    flush()
    return tuple(chunks)


def _split_jsonl(
    resource: SkillResource, text: str
) -> tuple[SkillChunk, ...]:
    chunks = []
    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {resource.path}:{line_no}: {exc}") from exc
        title = ""
        if isinstance(payload, dict):
            title = str(
                payload.get("id")
                or payload.get("character")
                or payload.get("scenario")
                or ""
            )
        chunks.append(
            SkillChunk(
                resource_id=resource.resource_id,
                path=resource.path,
                evidence_type=resource.evidence_type,
                authoritative=resource.authoritative,
                line_start=line_no,
                line_end=line_no,
                title=title,
                text=stripped,
            )
        )
    return tuple(chunks)


def load_skill_index(root: Path) -> SkillIndex:
    root = root.resolve()
    catalog_path = root / CATALOG_NAME
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("version") != 2:
        raise ValueError("resource_catalog.json must use version 2")
    raw_resources = catalog.get("resources")
    if not isinstance(raw_resources, list) or not raw_resources:
        raise ValueError("resource catalog has no resources")

    resources: dict[str, SkillResource] = {}
    declared_paths: set[str] = set()
    for item in raw_resources:
        if not isinstance(item, dict):
            raise ValueError("resource entry must be an object")
        resource_id = str(item.get("id") or "").strip()
        rel_path = str(item.get("path") or "").strip().replace("\\", "/")
        evidence_type = str(item.get("evidence_type") or "").strip()
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if not resource_id or resource_id in resources:
            raise ValueError(f"invalid or duplicate resource id: {resource_id!r}")
        if not rel_path or rel_path in declared_paths:
            raise ValueError(f"invalid or duplicate resource path: {rel_path!r}")
        if evidence_type not in {"fact", "boundary", "behavior_range", "example"}:
            raise ValueError(f"invalid evidence_type for {resource_id}")
        path = (root / rel_path).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"hash mismatch for {rel_path}: expected {expected_sha}, got {actual_sha}"
            )
        crosscheck = item.get("crosscheck") or []
        if not isinstance(crosscheck, list) or not all(
            isinstance(value, str) for value in crosscheck
        ):
            raise ValueError(f"invalid crosscheck for {resource_id}")
        resources[resource_id] = SkillResource(
            resource_id=resource_id,
            path=rel_path,
            scope=str(item.get("scope") or "").strip(),
            evidence_type=evidence_type,
            authoritative=bool(item.get("authoritative")),
            crosscheck=tuple(crosscheck),
            sha256=actual_sha,
        )
        declared_paths.add(rel_path)

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if _is_formal_content(path)
    }
    if declared_paths != actual_paths:
        missing = sorted(actual_paths - declared_paths)
        stale = sorted(declared_paths - actual_paths)
        raise ValueError(
            f"catalog coverage mismatch: missing={missing}, stale={stale}"
        )
    for resource in resources.values():
        unknown = set(resource.crosscheck) - set(resources)
        if unknown:
            raise ValueError(
                f"unknown crosscheck for {resource.resource_id}: {sorted(unknown)}"
            )

    chunks: dict[str, tuple[SkillChunk, ...]] = {}
    for resource in resources.values():
        path = root / resource.path
        text = path.read_text(encoding="utf-8")
        if "\ufffd" in text:
            raise ValueError(f"replacement character in {resource.path}")
        if path.suffix.casefold() == ".jsonl":
            resource_chunks = _split_jsonl(resource, text)
        else:
            resource_chunks = _split_markdown(resource, text)
        if not resource_chunks:
            raise ValueError(f"resource has no indexable content: {resource.path}")
        chunks[resource.resource_id] = resource_chunks
    return SkillIndex(
        root=root,
        signature=skill_signature(root),
        resources=resources,
        chunks=chunks,
    )


def ensure_skill_index(root: Path, previous: SkillIndex | None) -> SkillIndex:
    signature = skill_signature(root)
    if previous is not None and previous.signature == signature:
        return previous
    return load_skill_index(root)


def catalog_prompt(index: SkillIndex) -> str:
    rows = []
    for resource in index.resources.values():
        rows.append(
            f"{resource.resource_id}={resource.scope}；证据类型={resource.evidence_type}"
        )
    return "\n".join(rows)


def _compact(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value.casefold())


def _bigrams(value: str) -> set[str]:
    compact = _compact(value)
    if len(compact) < 2:
        return {compact} if compact else set()
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _score_chunk(chunk: SkillChunk, entities: list[str], concepts: list[str]) -> float:
    haystack = f"{chunk.title}\n{chunk.text}".casefold()
    compact_haystack = _compact(haystack)
    haystack_bigrams = _bigrams(compact_haystack)
    score = 0.0
    for entity in entities:
        folded = entity.casefold().strip()
        # A one-character Chinese address is useful for resolving the speaker,
        # but far too common to rank evidence. Let sentence-level concepts do
        # the ranking instead of promoting almost every paragraph equally.
        if len(_compact(folded)) < 2:
            continue
        if folded and folded in haystack:
            score += 14.0
        entity_bigrams = _bigrams(folded)
        if entity_bigrams:
            score += 4.0 * len(entity_bigrams & haystack_bigrams) / len(entity_bigrams)
    for concept in concepts:
        folded = concept.casefold().strip()
        if folded and folded in haystack:
            score += 7.0
        concept_bigrams = _bigrams(folded)
        if concept_bigrams:
            score += 3.0 * len(concept_bigrams & haystack_bigrams) / len(concept_bigrams)
    if chunk.title and any(
        value.casefold() in chunk.title.casefold()
        for value in [*entities, *concepts]
        if value
    ):
        score += 5.0
    return score


def _clean_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value[:limit]:
        if isinstance(item, str):
            item = item.strip()
            if 1 <= len(item) <= 48 and item not in result:
                result.append(item)
    return result


def normalize_requests(
    index: SkillIndex, requests: list[dict[str, Any]], source_text: str
) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for request in requests[:6]:
        if not isinstance(request, dict):
            continue
        resource_id = request.get("resource_id")
        if resource_id not in index.resources or resource_id in seen:
            continue
        entities = [
            item
            for item in _clean_list(request.get("entities"), 4)
            if item in source_text
        ]
        concepts = _clean_list(
            request.get("concepts", request.get("anchors")), 6
        )
        evidence_type = request.get("evidence_type")
        if evidence_type not in {"fact", "boundary", "behavior_range", "example"}:
            evidence_type = index.resources[resource_id].evidence_type
        if not entities and not concepts:
            continue
        normalized.append(
            {
                "resource_id": resource_id,
                "entities": entities,
                "concepts": concepts,
                "evidence_type": evidence_type,
            }
        )
        seen.add(resource_id)
    expanded = list(normalized)
    for request in list(normalized):
        for companion in index.resources[request["resource_id"]].crosscheck:
            if companion in seen or len(expanded) >= 4:
                continue
            copied = dict(request)
            copied["resource_id"] = companion
            copied["evidence_type"] = index.resources[companion].evidence_type
            expanded.append(copied)
            seen.add(companion)
    return expanded[:4]


def select_skill_chunks(
    index: SkillIndex,
    requests: list[dict[str, Any]],
    source_text: str,
    max_chars: int = 4500,
) -> tuple[list[SkillChunk], list[str]]:
    requests = normalize_requests(index, requests, source_text)
    selected: list[SkillChunk] = []
    missing: list[str] = []
    used_keys: set[tuple[str, int, int]] = set()
    used_chars = 0
    for request in requests:
        resource_id = request["resource_id"]
        candidates = [
            (_score_chunk(chunk, request["entities"], request["concepts"]), chunk)
            for chunk in index.chunks[resource_id]
        ]
        positive = [item for item in candidates if item[0] > 0]
        positive.sort(key=lambda item: (-item[0], item[1].line_start))
        max_chunks = 3 if request["evidence_type"] in {"behavior_range", "example"} else 2
        added = 0
        for _, chunk in positive:
            key = (chunk.path, chunk.line_start, chunk.line_end)
            if key in used_keys:
                continue
            projected = used_chars + len(chunk.text)
            if projected > max_chars:
                continue
            selected.append(chunk)
            used_keys.add(key)
            used_chars = projected
            added += 1
            if added >= max_chunks:
                break
        if added == 0:
            missing.append(resource_id)
    return selected, missing


def render_prefetch(
    index: SkillIndex,
    requests: list[dict[str, Any]],
    source_text: str,
    max_chars: int = 4500,
) -> tuple[str, bool, list[dict[str, Any]]] | None:
    normalized = normalize_requests(index, requests, source_text)
    if not normalized:
        return None
    chunks, missing = select_skill_chunks(index, normalized, source_text, max_chars)
    if not chunks:
        return None
    blocks = []
    trace = []
    for chunk in chunks:
        label = {
            "fact": "硬事实",
            "boundary": "知情或推导边界",
            "behavior_range": "行为范围",
            "example": "表达例子",
        }[chunk.evidence_type]
        blocks.append(
            f"[{label}｜{chunk.resource_id}]\n{chunk.text}"
        )
        trace.append(
            {
                "resource_id": chunk.resource_id,
                "path": chunk.path,
                "sha256": index.resources[chunk.resource_id].sha256,
                "line_start": chunk.line_start,
                "line_end": chunk.line_end,
                "evidence_type": chunk.evidence_type,
                "chars": len(chunk.text),
            }
        )
    complete = not missing
    coverage = (
        "所需依据已经完整提供，不再取得同一资料。"
        if complete
        else "现有依据不完整；只有缺失部分确实影响答案时才允许补查一次。"
    )
    header = (
        "以下是当前回复所需的权威 Skill 片段，不是当前场景中新发生的事实。"
        + coverage
        + "硬事实和知情边界必须遵守；行为范围只提供合理反应空间；"
        "表达例子只帮助理解声音，不得复读、套用其中的一次性场景事实或固定动作。"
        "当知情或证据边界要求区分明确事实、合理推断与未知时，"
        "只能用当前已提供的证据作说明；不得为了举例而新造外貌、能力、共同经历或生活习惯。"
        "在这些边界内，根据当前意图、关系、身体状态和场景自由组织一次自然中文回复。"
        "如果用户明确询问一个能够由硬事实直接回答的问题，必须先让结论清楚可辨；"
        "角色式反问、留白或玩笑可以随后出现，但不能代替事实答案。"
        "资料若区分默认或当前阶段与后续阶段，必须按当前结构化路线选择对应结论；"
        "情感上的亲近、默契或心照不宣不能被用来反转尚未成立的正式关系状态。"
        "询问客观状态时，要区分实际事实与人物可能对外采用的掩饰说法；"
        "社交口径不能改写事实，后续的玩笑、反话或潜台词也不能推翻本轮已经给出的明确结论。"
        "没有建立的近期细节可以作不冲突、无重要后果的日常即兴，以维持生活感；"
        "但不能用即兴内容建立或证明关系变化、重大决定、伤势、冲突、路线、承诺、"
        "隐藏事实或其他会持续影响后续的事件，也不能把旧经历改写成近期重大事实。"
        "不得说明资料来源、文件、检索、分类或内部处理过程。\n"
    )
    return header + "\n\n".join(blocks), complete, trace
