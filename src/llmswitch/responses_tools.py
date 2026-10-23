"""Typed built-in Responses tool request helpers."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _ResponsesTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def to_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class WebSearchTool(_ResponsesTool):
    """Built-in web-search tool configuration for ``responses.create``."""

    type: Literal["web_search"] = "web_search"
    filters: dict[str, Any] | None = None
    search_context_size: Literal["low", "medium", "high"] | None = None
    user_location: dict[str, Any] | None = None
    external_web_access: bool | None = None


class FileSearchTool(_ResponsesTool):
    """Built-in file-search tool configuration for ``responses.create``."""

    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str] = Field(min_length=1)
    filters: dict[str, Any] | None = None
    max_num_results: int | None = Field(default=None, gt=0)
    ranking_options: dict[str, Any] | None = None


class CodeInterpreterTool(_ResponsesTool):
    """Built-in code-interpreter tool configuration for ``responses.create``."""

    type: Literal["code_interpreter"] = "code_interpreter"
    container: str | dict[str, Any] | None = None


class ImageGenerationTool(_ResponsesTool):
    """Built-in image-generation tool configuration for ``responses.create``."""

    type: Literal["image_generation"] = "image_generation"
    size: str | None = None
    quality: str | None = None
    format: str | None = None
    background: str | None = None
    moderation: str | None = None
    output_compression: int | None = Field(default=None, ge=0, le=100)
    partial_images: int | None = Field(default=None, ge=0)


class ComputerTool(_ResponsesTool):
    """Built-in computer-use tool configuration for ``responses.create``.

    This model only describes the provider tool. LLMSwitch never executes a
    returned computer action; application code must make that approval decision.
    """

    type: Literal["computer"] = "computer"


class McpTool(_ResponsesTool):
    """Remote MCP server configuration; switch policy must explicitly allow it."""

    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str | None = None
    connector_id: str | None = None
    authorization: str | None = None
    require_approval: Any = "always"

class ConnectorTool(McpTool):
    """OpenAI connector configuration carried through the MCP tool transport."""

    connector_id: str


class McpApprovalDecision(BaseModel):
    """Application-controlled approval or rejection of a pending MCP call."""

    model_config = ConfigDict(extra="forbid")

    approval_request_id: str
    approve: bool

    def to_input_item(self) -> dict[str, Any]:
        return {
            "type": "mcp_approval_response",
            "approval_request_id": self.approval_request_id,
            "approve": self.approve,
        }
