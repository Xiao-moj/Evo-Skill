"""
Base class for agent CLIs installed and executed inside Docker.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Environment as JinjaEnv

from .agent_context import AgentContext
from .docker_environment import DockerEnvironment


@dataclass
class ExecInput:
    """One command to run inside the container."""

    command: str
    cwd: Optional[str] = None
    env: Optional[Dict[str, str]] = None
    timeout_sec: Optional[int] = None


class BaseInstalledAgent(ABC):
    """Common lifecycle for Docker-installed agent CLIs."""

    SUPPORTS_TRAJECTORY: bool = False

    def __init__(
        self,
        logs_dir: Path,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        **_: object,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.model_name = model_name
        self._version = version
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    @abstractmethod
    def name() -> str:
        """Returns the agent name."""

    @property
    @abstractmethod
    def _install_script_template_path(self) -> Path:
        """Returns the shell-template path used to install the CLI."""

    @abstractmethod
    def create_run_commands(self, instruction: str) -> List[ExecInput]:
        """Builds the command sequence used to run the CLI."""

    @abstractmethod
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Parses logs and fills the run context."""

    def version(self) -> Optional[str]:
        """Returns the requested CLI version."""

        return self._version

    @property
    def _template_variables(self) -> Dict[str, str]:
        """Template variables exposed to install scripts."""

        variables: Dict[str, str] = {}
        if self.version():
            variables["version"] = str(self.version())
        if self.model_name:
            variables["model_name"] = str(self.model_name)
        return variables

    async def setup(self, environment: DockerEnvironment) -> None:
        """Installs the agent CLI inside the container."""

        print(f"[{self.name()}] Setting up agent in container...")
        result = await environment.exec("mkdir -p /installed-agent")
        if result.return_code != 0:
            raise RuntimeError("Failed to create /installed-agent directory")

        if not self._install_script_template_path.exists():
            raise FileNotFoundError(
                f"Install script template not found: {self._install_script_template_path}"
            )

        template = JinjaEnv().from_string(self._install_script_template_path.read_text())
        rendered_script = template.render(**self._template_variables)

        script_path = self.logs_dir / "install.sh"
        script_path.write_text(rendered_script)
        print(f"[{self.name()}] Install script saved to: {script_path}")

        await environment.upload_file(
            source_path=script_path,
            target_path="/installed-agent/install.sh",
        )

        print(f"[{self.name()}] Running install script...")
        result = await environment.exec(
            command="bash /installed-agent/install.sh",
            timeout_sec=600,
        )

        setup_dir = self.logs_dir / "setup"
        setup_dir.mkdir(parents=True, exist_ok=True)
        (setup_dir / "return-code.txt").write_text(str(result.return_code))
        if result.stdout:
            (setup_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr:
            (setup_dir / "stderr.txt").write_text(result.stderr)

        if result.return_code != 0:
            raise RuntimeError(
                f"Agent setup failed with exit code {result.return_code}. See logs in {setup_dir}"
            )

        print(f"[{self.name()}] Agent setup completed successfully")

    async def run(
        self,
        instruction: str,
        environment: DockerEnvironment,
        context: AgentContext,
    ) -> None:
        """Runs the agent CLI inside the container."""

        print(f"[{self.name()}] Running agent with instruction: {instruction[:100]}...")
        commands = self.create_run_commands(instruction)
        for i, exec_input in enumerate(commands):
            print(f"[{self.name()}] Executing command {i + 1}/{len(commands)}...")
            command_dir = self.logs_dir / f"command-{i}"
            command_dir.mkdir(parents=True, exist_ok=True)
            (command_dir / "command.txt").write_text(exec_input.command)

            result = await environment.exec(
                command=exec_input.command,
                cwd=exec_input.cwd,
                env=exec_input.env,
                timeout_sec=exec_input.timeout_sec,
            )
            (command_dir / "return-code.txt").write_text(str(result.return_code))
            if result.stdout:
                (command_dir / "stdout.txt").write_text(result.stdout)
            if result.stderr:
                (command_dir / "stderr.txt").write_text(result.stderr)
            print(
                f"[{self.name()}] Command {i + 1} completed with return code: {result.return_code}"
            )

        print(f"[{self.name()}] Parsing execution results...")
        self.populate_context_post_run(context)
