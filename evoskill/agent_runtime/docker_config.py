"""Docker image selection helpers for agent runtimes."""

from __future__ import annotations

from typing import Optional


class DockerConfig:
    """Manages whether EvoSkill uses a prebuilt agents image."""

    USE_PREBUILT_IMAGE = False
    PREBUILT_IMAGE_NAME = "evoskill-agents:latest"
    DEFAULT_IMAGE_NAME = "python:3.11-slim"

    @classmethod
    def get_image_name(cls) -> str:
        """Returns the configured image name."""

        if cls.USE_PREBUILT_IMAGE:
            return cls.PREBUILT_IMAGE_NAME
        return cls.DEFAULT_IMAGE_NAME

    @classmethod
    def should_skip_installation(cls) -> bool:
        """Returns True when the image already contains the agent CLIs."""

        return cls.USE_PREBUILT_IMAGE

    @classmethod
    def enable_prebuilt_image(cls, image_name: Optional[str] = None) -> None:
        """Enables prebuilt-image mode."""

        cls.USE_PREBUILT_IMAGE = True
        if image_name:
            cls.PREBUILT_IMAGE_NAME = str(image_name)

    @classmethod
    def disable_prebuilt_image(cls) -> None:
        """Disables prebuilt-image mode."""

        cls.USE_PREBUILT_IMAGE = False
