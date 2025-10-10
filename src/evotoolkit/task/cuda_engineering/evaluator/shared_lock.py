# Copyright (c) 2025 Ping Guo
# Licensed under the MIT License


# 文件名：shared_lock.py
import os
import time
import tempfile
import psutil
import json
from contextlib import contextmanager
from typing import Optional

# 使用 portalocker 库（跨平台）
import portalocker

LOCK_FILE = os.path.join(tempfile.gettempdir(), "evotool_cross_process.lock")
LOCK_INFO_FILE = os.path.join(tempfile.gettempdir(), "evotool_lock_info.json")


def _is_process_alive(pid: int) -> bool:
    """检查进程是否还活着"""
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return False


def _cleanup_stale_lock():
    """清理已死进程或长时间卡住的锁"""
    if not os.path.exists(LOCK_INFO_FILE):
        return

    try:
        with open(LOCK_INFO_FILE, "r") as f:
            lock_info = json.load(f)

        pid = lock_info.get("pid")
        timestamp = lock_info.get("timestamp", 0)
        current_time = time.time()

        should_cleanup = False

        if pid and not _is_process_alive(pid):
            # 进程已死，清理锁文件
            should_cleanup = True
        elif pid and (current_time - timestamp) > 1800:  # 30分钟超时
            # 进程卡住超过30分钟，强制清理
            try:
                # 检查是否为孤儿进程 (PPID=1)
                process = psutil.Process(pid)
                if process.ppid() == 1:  # 孤儿进程
                    process.kill()  # 强制杀死孤儿进程
                    should_cleanup = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                should_cleanup = True

        if should_cleanup:
            for file_path in [LOCK_FILE, LOCK_INFO_FILE]:
                try:
                    os.remove(file_path)
                except OSError:
                    pass

    except Exception:
        # 如果读取失败，也尝试清理
        for file_path in [LOCK_FILE, LOCK_INFO_FILE]:
            try:
                os.remove(file_path)
            except OSError:
                pass


@contextmanager
def global_file_lock(timeout: Optional[float] = None):
    """全局文件锁，支持超时和死锁检测

    Args:
        timeout: 超时时间（秒），None表示无限等待
    """
    # 清理可能的死锁
    _cleanup_stale_lock()

    start_time = time.time()
    current_pid = os.getpid()

    while True:
        try:
            # 尝试获取锁
            with open(LOCK_FILE, "w") as f:
                portalocker.lock(f, portalocker.LOCK_EX | portalocker.LOCK_NB)

                # 记录锁信息
                lock_info = {"pid": current_pid, "timestamp": time.time()}
                with open(LOCK_INFO_FILE, "w") as info_f:
                    json.dump(lock_info, info_f)

                try:
                    yield
                finally:
                    portalocker.unlock(f)
                    # 清理锁信息
                    try:
                        os.remove(LOCK_INFO_FILE)
                    except OSError:
                        pass
                break

        except portalocker.LockException:
            # 锁被占用，检查超时
            if timeout is not None and (time.time() - start_time) >= timeout:
                raise TimeoutError(f"Failed to acquire lock within {timeout} seconds")

            # 等待一会儿再试
            time.sleep(0.1)

            # 检查是否需要清理死锁
            if (time.time() - start_time) % 10 < 0.1:  # 每10秒检查一次
                _cleanup_stale_lock()
