#!/usr/bin/env python3
"""
FastAPI Web 服务入口
提供：
1. / 静态网页前端
2. /api/scan 扫描服务器目录（支持按文件夹聚合、已有SRT标记、多选过滤）
3. /api/tasks/server-batch 批量转录已勾选的服务器本地视频
4. /api/tasks/upload 网页上传文件转录（支持独立文件夹存储）
5. /api/tasks 历史任务列表
6. /api/tasks/{id} 任务实时/历史状态查询（支持刷新恢复）
7. /api/tasks/{id}/events SSE 实时进度推流
8. /api/tasks/{id}/download-all 一键打包下载所有 SRT (tar.gz)
9. /api/download/{id}/{index}/{type} 下载单个文件 (SRT / TXT / JSON)
10. /api/status GPU 显存与引擎状态
"""

import asyncio
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .corrector import get_llm_status, is_llm_available, process_json_file_standalone
from .engine import engine
from .task_queue import (
    BASE_DIR,
    DATA_DIR,
    MEDIA_EXTENSIONS,
    TASKS_DIR,
    get_media_duration,
    task_manager,
)

WEB_DIR = BASE_DIR / "web"
IGNORED_DIRS = {
    ".venv", ".git", "__pycache__", "node_modules", ".cache",
    ".gemini", ".system_generated", "@eaDir", "$RECYCLE.BIN",
    "System Volume Information", ".DS_Store",
}

app = FastAPI(title="Qwen3-ASR High-Throughput Subtitle Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """启动时异步预加载模型"""
    asyncio.get_event_loop().run_in_executor(None, engine.load_model)


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Web index.html not found")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    gpu = engine.get_gpu_status()
    llm_info = get_llm_status()
    return {
        "is_loaded": engine.is_loaded,
        "model_name": engine.model_name,
        "aligner_name": engine.aligner_name,
        "llm_ready": llm_info["available"],
        "llm_model": llm_info["model"] if llm_info["available"] else "",
        "llm_models": llm_info["models"],
        "gpu": gpu,
    }


class ScanRequest(BaseModel):
    path: str
    recursive: bool = True


@app.post("/api/scan")
async def scan_directory(req: ScanRequest):
    raw_path = (req.path or "").strip().strip('"\'')
    if not raw_path:
        raise HTTPException(status_code=400, detail="指定路径不能为空")
    root = Path(raw_path).expanduser()
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"指定路径不存在: {req.path}")
    if root.is_symlink():
        try:
            root = root.resolve()
        except Exception:
            pass

    def should_ignore(p: Path) -> bool:
        for part in p.parts:
            if part in IGNORED_DIRS or part.startswith(".") or part == "@eaDir":
                return True
        return False

    if root.is_file():
        candidates = [root]
    elif req.recursive:
        candidates = sorted(root.rglob("*"))
    else:
        candidates = sorted(root.glob("*"))

    files_list = []
    folders_dict: Dict[str, Dict[str, Any]] = {}
    seen = set()

    for f in candidates:
        if not f.is_file():
            continue
        if f.suffix.lower() not in MEDIA_EXTENSIONS:
            continue
        try:
            rel_part = f.relative_to(root if root.is_dir() else root.parent)
        except Exception:
            rel_part = Path(f.name)
        if should_ignore(rel_part):
            continue
        try:
            resolved = f.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        dur = get_media_duration(str(resolved)) or 0.0
        parent_dir = resolved.parent
        has_srt = resolved.with_suffix(".srt").exists() or (parent_dir / "srt" / f"{resolved.stem}.srt").exists()

        try:
            rel_folder = parent_dir.relative_to(root if root.is_dir() else root.parent)
            if str(rel_folder) == ".":
                folder_display = (root.name or str(root)) + " (根目录)"
            else:
                folder_display = str(rel_folder)
        except Exception:
            folder_display = parent_dir.name or str(parent_dir)

        parent_folder_path = str(parent_dir)

        file_info = {
            "path": str(resolved),
            "name": resolved.name,
            "folder": folder_display,
            "folder_path": parent_folder_path,
            "size_mb": round(resolved.stat().st_size / (1024 * 1024), 2),
            "duration": dur,
            "has_srt": has_srt,
        }
        files_list.append(file_info)

        if parent_folder_path not in folders_dict:
            folders_dict[parent_folder_path] = {
                "folder_path": parent_folder_path,
                "folder_name": folder_display,
                "count": 0,
                "duration_sec": 0.0,
                "files": [],
            }
        folders_dict[parent_folder_path]["count"] += 1
        folders_dict[parent_folder_path]["duration_sec"] += dur
        folders_dict[parent_folder_path]["files"].append(file_info)

    folders_list = list(folders_dict.values())
    folders_list.sort(key=lambda x: x["folder_name"])
    for idx, f in enumerate(folders_list):
        f["folder_id"] = f"fg_{idx}"
        f["duration_sec"] = round(f["duration_sec"], 2)

    total_duration = sum(x["duration"] for x in files_list)

    return {
        "root_path": str(root),
        "count": len(files_list),
        "total_duration_sec": round(total_duration, 2),
        "total_duration_min": round(total_duration / 60.0, 1),
        "folders": folders_list,
        "files": files_list,
    }


class ServerBatchRequest(BaseModel):
    selected_paths: Optional[List[str]] = None
    path: Optional[str] = None
    recursive: bool = True
    force: bool = True
    srt_folder: bool = False
    language: Optional[str] = None
    max_duration: float = 6.0
    gap_threshold: float = 0.3
    concurrency: int = 5
    enable_llm: bool = True


def _peek_json_transcript(path: Path) -> Optional[Dict[str, Any]]:
    """读取 ASR JSON 的元信息；不是有效的转录文件时返回 None"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    segments: Optional[List[Any]] = None
    domain = ""
    if isinstance(data, list):
        segments = data
    elif isinstance(data, dict):
        segments = (
            data.get("segments")
            or data.get("sentences")
            or data.get("utterances")
            or data.get("word_timestamps")
            or data.get("words")
            or data.get("tokens")
        )
        domain = (data.get("domain_profile") or {}).get("domain", "")
    else:
        return None

    if not isinstance(segments, list) or not segments:
        return None

    duration = 0.0
    try:
        last = segments[-1]
        if isinstance(last, dict):
            duration = float(
                last.get("end_time")
                or last.get("end")
                or last.get("orig_end")
                or 0.0
            )
    except Exception:
        duration = 0.0

    return {
        "segments_count": len(segments),
        "duration": duration,
        "domain": domain,
    }


class ScanJsonRequest(BaseModel):
    path: str
    recursive: bool = True


@app.post("/api/scan-json")
async def scan_json_directory(req: ScanJsonRequest):
    """扫描服务器目录下已有的 ASR JSON 转录文件（用于批量大模型纠错）"""
    raw_path = (req.path or "").strip().strip('"\'')
    if not raw_path:
        raise HTTPException(status_code=400, detail="指定路径不能为空")
    root = Path(raw_path).expanduser()
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"指定路径不存在: {req.path}")
    if root.is_symlink():
        try:
            root = root.resolve()
        except Exception:
            pass

    if root.is_file():
        candidates = [root]
    elif req.recursive:
        candidates = sorted(root.rglob("*.json"))
    else:
        candidates = sorted(root.glob("*.json"))

    def ignored(p: Path) -> bool:
        for part in p.parts:
            if part in IGNORED_DIRS or part.startswith(".") or part == "@eaDir":
                return True
        if p.name == "task.json":
            return True
        return False

    files_list: List[Dict[str, Any]] = []
    folders_dict: Dict[str, Dict[str, Any]] = {}
    seen = set()

    for f in candidates:
        if not f.is_file():
            continue
        try:
            rel_part = f.relative_to(root if root.is_dir() else root.parent)
        except Exception:
            rel_part = Path(f.name)
        if ignored(rel_part):
            continue
        try:
            resolved = f.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        meta = _peek_json_transcript(resolved)
        if meta is None:
            continue

        seen.add(resolved)
        parent = resolved.parent
        has_srt = resolved.with_suffix(".srt").exists() or (parent / "srt" / f"{resolved.stem}.srt").exists()

        try:
            rel_folder = parent.relative_to(root if root.is_dir() else root.parent)
            if str(rel_folder) == ".":
                folder_display = (root.name or str(root)) + " (根目录)"
            else:
                folder_display = str(rel_folder)
        except Exception:
            folder_display = parent.name or str(parent)

        parent_folder_path = str(parent)
        file_info = {
            "path": str(resolved),
            "name": resolved.name,
            "folder": folder_display,
            "folder_path": parent_folder_path,
            "size_mb": round(resolved.stat().st_size / (1024 * 1024), 2),
            "segments_count": meta["segments_count"],
            "duration": meta["duration"],
            "domain": meta["domain"],
            "has_srt": has_srt,
            "modified": resolved.stat().st_mtime,
        }
        files_list.append(file_info)

        folder_entry = folders_dict.setdefault(
            parent_folder_path,
            {
                "folder_path": parent_folder_path,
                "folder_name": folder_display,
                "count": 0,
                "segments_total": 0,
                "duration_sec": 0.0,
                "files": [],
            },
        )
        folder_entry["count"] += 1
        folder_entry["segments_total"] += meta["segments_count"]
        folder_entry["duration_sec"] += meta["duration"]
        folder_entry["files"].append(file_info)

    folders_list = sorted(folders_dict.values(), key=lambda x: x["folder_name"])
    for idx, f in enumerate(folders_list):
        f["folder_id"] = f"fg_json_{idx}"
        f["duration_sec"] = round(f["duration_sec"], 2)

    return {
        "root_path": str(root),
        "count": len(files_list),
        "total_segments": sum(f["segments_count"] for f in files_list),
        "total_duration_min": round(sum(f["duration"] for f in files_list) / 60.0, 1),
        "folders": folders_list,
        "files": files_list,
    }


class JsonCorrectionTaskRequest(BaseModel):
    selected_paths: Optional[List[str]] = None
    path: Optional[str] = None
    recursive: bool = True
    batch_size: int = 25
    concurrency: int = 5
    llm_model: str = "auto"


@app.post("/api/tasks/json-correction")
async def create_json_correction_task(req: JsonCorrectionTaskRequest):
    """创建「服务器已有 JSON 文件大模型纠错」任务，进度在看板实时展示"""
    target_paths: List[str] = []
    if req.selected_paths and len(req.selected_paths) > 0:
        target_paths = [p for p in req.selected_paths if Path(p).exists()]
    elif req.path:
        root = Path(req.path)
        if not root.exists():
            raise HTTPException(status_code=400, detail="指定路径不存在")
        cands = sorted(root.rglob("*.json")) if req.recursive else sorted(root.glob("*.json"))
        target_paths = [str(f.resolve()) for f in cands if f.is_file()]

    files_data: List[Dict[str, Any]] = []
    seen = set()
    for p in target_paths:
        path_obj = Path(p)
        if p in seen:
            continue
        seen.add(p)
        meta = _peek_json_transcript(path_obj)
        if meta is None:
            continue
        files_data.append({
            "path": str(path_obj),
            "name": path_obj.name,
            "duration": meta["duration"],
            "segments_count": meta["segments_count"],
        })

    if not files_data:
        raise HTTPException(status_code=400, detail="没有选择任何有效的 ASR JSON 文件")

    params = {
        "batch_size": req.batch_size,
        "concurrency": req.concurrency,
        "llm_model": req.llm_model,
        "llm_api_base": "http://127.0.0.1:8002/v1",
    }
    task = task_manager.create_task("json_correction", files_data, params)
    task_manager.start_task(task.task_id)

    return {
        "task_id": task.task_id,
        "total_files": len(files_data),
        "total_segments": sum(f["segments_count"] for f in files_data),
    }


@app.post("/api/tasks/server-batch")
async def create_server_batch_task(req: ServerBatchRequest):
    target_paths: List[str] = []

    if req.selected_paths and len(req.selected_paths) > 0:
        target_paths = [p for p in req.selected_paths if Path(p).exists()]
    elif req.path:
        root = Path(req.path)
        if not root.exists():
            raise HTTPException(status_code=400, detail="指定路径不存在")
        cands = sorted(root.rglob("*")) if req.recursive else sorted(root.glob("*"))
        for f in cands:
            if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
                target_paths.append(str(f.resolve()))

    if not target_paths:
        raise HTTPException(status_code=400, detail="没有选择任何有效的视频文件")

    files_data = []
    for p in target_paths:
        fpath = Path(p)
        dur = get_media_duration(str(fpath)) or 0.0
        files_data.append({
            "path": str(fpath),
            "name": fpath.name,
            "duration": dur,
        })

    params = {
        "force": req.force,
        "srt_folder": req.srt_folder,
        "language": req.language,
        "max_duration": req.max_duration,
        "gap_threshold": req.gap_threshold,
        "concurrency": req.concurrency,
        "enable_llm": req.enable_llm,
    }

    task = task_manager.create_task("server_batch", files_data, params)
    task_manager.start_task(task.task_id)

    return {
        "task_id": task.task_id,
        "total_files": len(files_data),
        "total_duration": task.total_audio_duration,
    }


@app.post("/api/tasks/upload")
async def upload_files_for_transcription(
    files: List[UploadFile] = File(...),
    language: Optional[str] = Form(None),
    max_duration: float = Form(6.0),
    gap_threshold: float = Form(0.3),
    concurrency: int = Form(5),
    enable_llm: bool = Form(True),
):

    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件")

    task_id = str(uuid.uuid4())[:8]
    task_dir = TASKS_DIR / task_id
    inputs_dir = task_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "outputs").mkdir(parents=True, exist_ok=True)

    saved_items = []
    for f in files:
        safe_name = Path(f.filename).name
        out_path = inputs_dir / safe_name
        with open(out_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        dur = get_media_duration(str(out_path)) or 0.0
        saved_items.append({
            "path": str(out_path),
            "name": safe_name,
            "duration": dur,
        })

    params = {
        "force": True,
        "srt_folder": False,
        "language": language,
        "max_duration": max_duration,
        "gap_threshold": gap_threshold,
        "concurrency": concurrency,
        "enable_llm": enable_llm,
    }

    # 创建并启动任务
    task = task_manager.create_task("upload", saved_items, params, task_id=task_id)
    task_manager.start_task(task_id)

    return {
        "task_id": task_id,
        "total_files": len(saved_items),
        "total_duration": task.total_audio_duration,
    }


@app.post("/api/tasks/{task_id}/cancel")
async def cancel_task_endpoint(task_id: str):
    """强制取消 / 终止任务"""
    success = task_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="任务不存在或已结束")
    return {"success": True, "message": "任务已成功请求终止"}


@app.delete("/api/tasks/{task_id}")
async def delete_task_endpoint(task_id: str, delete_files: bool = True):
    """删除指定历史任务记录，并清理相关磁盘文件"""
    success = task_manager.delete_task(task_id, delete_files=delete_files)
    return {"success": success, "message": "任务记录已删除"}


@app.post("/api/tasks/clear-history")
async def clear_history_endpoint(delete_files: bool = True):
    """一键清空所有历史任务并释放垃圾磁盘空间"""
    count = task_manager.clear_history(delete_files=delete_files)
    return {"success": True, "deleted_count": count, "message": f"已清理 {count} 个历史任务及临时文件"}


@app.get("/api/tasks")
async def list_recent_tasks():
    """获取历史任务简表"""
    return task_manager.list_tasks(limit=30)


@app.get("/api/tasks/{task_id}")
async def get_task_detail(task_id: str):
    """获取具体任务状态与文件列表（支持页面刷新后恢复）"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task



@app.get("/api/tasks/{task_id}/events")
async def task_events_sse(task_id: str):
    """SSE 实时推送任务执行进度、当前处理视频与实时速率"""
    task = task_manager.tasks.get(task_id)
    if not task:
        persisted = task_manager.get_task(task_id)
        if not persisted:
            raise HTTPException(status_code=404, detail="任务不存在")

    q = task_manager.subscribe(task_id)

    async def event_generator():
        try:
            # 立即推送一次当前最新状态
            cur = task_manager.get_task(task_id)
            if cur:
                import json
                yield f"data: {json.dumps(cur, ensure_ascii=False)}\n\n"

            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            task_manager.unsubscribe(task_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/tasks/{task_id}/download-all")
async def download_all_subtitles(task_id: str):
    """打包下载任务生成的所有 .srt 为 tar.gz 压缩包"""
    tar_path = task_manager.package_task_subtitles(task_id)
    if not tar_path or not tar_path.exists():
        raise HTTPException(status_code=404, detail="该任务尚未生成任何字幕文件")

    return FileResponse(
        str(tar_path),
        filename=f"{task_id}_all_subtitles.tar.gz",
        media_type="application/gzip",
    )


@app.get("/api/download/{task_id}/{file_index}/{export_type}")
async def download_file(task_id: str, file_index: int, export_type: str):
    """下载指定文件结果 (srt, txt, json)"""
    task = task_manager.tasks.get(task_id)
    if not task:
        data = task_manager.get_task(task_id)
        if not data or file_index >= len(data.get("files", [])):
            raise HTTPException(status_code=404, detail="文件不存在")
        file_item = data["files"][file_index]
    else:
        if file_index >= len(task.files):
            raise HTTPException(status_code=404, detail="文件索引越界")
        from dataclasses import asdict
        file_item = asdict(task.files[file_index])

    export_type = export_type.lower()
    key = f"{export_type}_path"
    target_path = file_item.get(key)

    # 兜底查找
    if not target_path or not Path(target_path).exists():
        stem = Path(file_item["name"]).stem
        alt_path = TASKS_DIR / task_id / "outputs" / f"{stem}.{export_type}"
        if alt_path.exists():
            target_path = str(alt_path)

    if not target_path or not Path(target_path).exists():
        raise HTTPException(status_code=404, detail=f"{export_type.upper()} 文件尚未生成或不存在")

    p = Path(target_path)
    content_types = {
        "srt": "text/plain; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }

    return FileResponse(
        str(p),
        filename=p.name,
        media_type=content_types.get(export_type, "application/octet-stream"),
    )


class CorrectJsonRequest(BaseModel):
    path: str
    output_srt: Optional[str] = None
    output_txt: Optional[str] = None


@app.post("/api/correct-json")
async def correct_json_server_file(req: CorrectJsonRequest):
    """针对服务器本地已有的 JSON 执行大模型智能纠错，并覆盖原 SRT/TXT/JSON"""
    target_path = Path(req.path)
    if not target_path.exists():
        raise HTTPException(status_code=400, detail=f"指定JSON文件不存在: {req.path}")

    try:
        res = await asyncio.get_event_loop().run_in_executor(
            None,
            process_json_file_standalone,
            str(target_path),
            req.output_srt,
            req.output_txt,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON纠错处理失败: {str(e)}")


@app.post("/api/correct-json-upload")
async def correct_json_uploaded_file(file: UploadFile = File(...)):
    """用户上传已有 JSON 文件进行智能纠错，生成并返回可下载的纠错字幕"""
    if not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="请上传标准 .json 格式的 ASR 识别文件")

    temp_id = f"corr_{uuid.uuid4().hex[:8]}"
    temp_dir = TASKS_DIR / temp_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    json_path = temp_dir / Path(file.filename).name
    with open(json_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        res = await asyncio.get_event_loop().run_in_executor(
            None,
            process_json_file_standalone,
            str(json_path),
        )
        res["temp_id"] = temp_id
        res["download_srt"] = f"/api/download-temp/{temp_id}/srt"
        res["download_txt"] = f"/api/download-temp/{temp_id}/txt"
        res["download_json"] = f"/api/download-temp/{temp_id}/json"
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON纠错处理失败: {str(e)}")


@app.get("/api/download-temp/{temp_id}/{export_type}")
async def download_temp_result(temp_id: str, export_type: str):
    """下载独立纠错任务生成的临时文件"""
    temp_dir = TASKS_DIR / temp_id
    if not temp_dir.exists():
        raise HTTPException(status_code=404, detail="临时任务已过期或不存在")

    matched = list(temp_dir.glob(f"*.{export_type.lower()}"))
    if not matched:
        raise HTTPException(status_code=404, detail=f"未找到对应的 {export_type.upper()} 文件")

    p = matched[0]
    content_types = {
        "srt": "text/plain; charset=utf-8",
        "txt": "text/plain; charset=utf-8",
        "json": "application/json; charset=utf-8",
    }
    return FileResponse(
        str(p),
        filename=p.name,
        media_type=content_types.get(export_type.lower(), "application/octet-stream"),
    )
