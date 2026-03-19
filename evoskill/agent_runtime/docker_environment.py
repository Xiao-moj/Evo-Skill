"""
Docker environment management for containerized agent CLIs.
"""

from __future__ import annotations

import asyncio
import asyncio.subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class ExecResult:
    """One command execution result."""

    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: int = 0


class DockerEnvironment:
    """Creates and manages one isolated Docker container for an agent session."""

    def __init__(
        self,
        session_id: str,
        work_dir: Path,
        agent_logs_dir: Path,
        skills_dir: Optional[Path] = None,
        image_name: str = "python:3.11-slim",
        docker_context: Optional[str] = None,
        memory: str = "2G",
        cpus: int = 2,
        network_mode: str = "bridge",
    ) -> None:
        self.session_id = str(session_id).lower().replace("_", "-").replace(".", "-")
        self.work_dir = Path(work_dir).resolve()
        self.agent_logs_dir = Path(agent_logs_dir).resolve()
        self.skills_dir = Path(skills_dir).resolve() if skills_dir else None
        self.image_name = str(image_name or "python:3.11-slim")
        self.docker_context = str(docker_context or "").strip()
        self.memory = str(memory or "2G")
        self.cpus = int(cpus or 2)
        self.network_mode = str(network_mode or "bridge")

        self._container_name = f"evoskill__{self.session_id}"
        self._is_started = False

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.agent_logs_dir.mkdir(parents=True, exist_ok=True)

    async def start(self) -> None:
        """Starts the Docker container."""

        print(f"[DockerEnvironment] Starting container: {self._container_name}")
        await self._cleanup_existing_container()
        cmd = self._build_docker_run_command()
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Failed to start container: {error_msg}")

        self._is_started = True
        print(f"[DockerEnvironment] Container started: {self._container_name}")
        await asyncio.sleep(1)

    def _build_docker_run_command(self) -> list[str]:
        """Builds the `docker run` command."""

        cmd = self._docker_base_command() + [
            "run",
            "-d",
            "--name",
            self._container_name,
            "--memory",
            self.memory,
            "--cpus",
            str(self.cpus),
            "--network",
            self.network_mode,
            "-v",
            f"{self.work_dir}:/workspace",
            "-v",
            f"{self.agent_logs_dir}:/logs/agent",
        ]
        if self.skills_dir and self.skills_dir.exists():
            cmd.extend(["-v", f"{self.skills_dir}:/root/.claude/skills:ro"])
            cmd.extend(["-v", f"{self.skills_dir}:/skills:ro"])
        cmd.extend(["-w", "/workspace", self.image_name, "tail", "-f", "/dev/null"])
        return cmd

    def _docker_base_command(self) -> list[str]:
        """Builds the base docker command with an optional explicit context."""

        cmd = ["docker"]
        if self.docker_context:
            cmd.extend(["--context", self.docker_context])
        return cmd

    async def _cleanup_existing_container(self) -> None:
        """Stops and removes a stale container with the same name."""

        process = await asyncio.create_subprocess_exec(
            *self._docker_base_command(),
            "rm",
            "-f",
            self._container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.communicate()

    async def exec(
        self,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> ExecResult:
        """Executes one command inside the container."""

        if not self._is_started:
            raise RuntimeError("Container not started")

        cmd = self._docker_base_command() + ["exec"]
        if cwd:
            cmd.extend(["-w", str(cwd)])
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.append(self._container_name)
        cmd.extend(["bash", "-c", command])

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            if timeout_sec:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            else:
                stdout, stderr = await process.communicate()
            return ExecResult(
                stdout=stdout.decode(errors="replace") if stdout else None,
                stderr=stderr.decode(errors="replace") if stderr else None,
                return_code=process.returncode or 0,
            )
        except asyncio.TimeoutError:
            try:
                process.kill()
                await process.communicate()
            except Exception:
                pass
            return ExecResult(
                stdout=None,
                stderr=f"Command timed out after {timeout_sec} seconds",
                return_code=-1,
            )
        except Exception as e:
            return ExecResult(stdout=None, stderr=str(e), return_code=-1)

    async def upload_file(self, source_path: Path, target_path: str) -> None:
        """Copies one local file into the container."""

        if not self._is_started:
            raise RuntimeError("Container not started")
        if not Path(source_path).exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        process = await asyncio.create_subprocess_exec(
            *self._docker_base_command(),
            "cp",
            str(source_path),
            f"{self._container_name}:{target_path}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            raise RuntimeError(f"Failed to upload file: {error_msg}")

    async def stop(self, keep_container: bool = False) -> None:
        """Stops the container and optionally removes it."""

        if not self._is_started:
            return

        print(f"[DockerEnvironment] Stopping container: {self._container_name}")
        stop_process = await asyncio.create_subprocess_exec(
            *self._docker_base_command(),
            "stop",
            self._container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await stop_process.communicate()

        if not keep_container:
            rm_process = await asyncio.create_subprocess_exec(
                *self._docker_base_command(),
                "rm",
                self._container_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await rm_process.communicate()
            print(f"[DockerEnvironment] Container removed: {self._container_name}")
        else:
            print(f"[DockerEnvironment] Container kept: {self._container_name}")

        self._is_started = False

    async def get_logs(self) -> str:
        """Returns `docker logs` for the live container."""

        if not self._is_started:
            return ""
        process = await asyncio.create_subprocess_exec(
            *self._docker_base_command(),
            "logs",
            self._container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        return stdout.decode(errors="replace") if stdout else ""
