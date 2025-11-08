"""Streaming chat utilities with Model Context Protocol integration."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Union

import httpx
from pydantic import BaseModel, Field

from mcp_client import mcp_manager

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], None] = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    pdf: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class ChatRequest(BaseModel):
    model_id: str
    chat_history: List[Message]
    use_mcp: bool = False
    approved_tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    mcp_auto_approve: bool = False


class ChatService:
    """High-level helper that orchestrates OpenRouter streaming and MCP tools."""

    def __init__(self, model_id: str, model_data: Dict[str, Any]) -> None:
        self.model_id = model_id
        self.model_data = model_data

    def prepare_messages(self, history: List[Message]) -> List[Dict[str, Any]]:
        prepared: List[Dict[str, Any]] = []
        for msg in history:
            entry: Dict[str, Any] = {"role": msg.role}
            content = msg.content
            if isinstance(content, list):
                entry["content"] = content
            elif isinstance(content, dict):
                entry["content"] = [content]
            elif content is None:
                entry["content"] = ""
            else:
                entry["content"] = str(content)
            if msg.role == "tool":
                if msg.tool_call_id:
                    entry["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    entry["name"] = msg.name
            prepared.append(entry)
        return prepared

    async def create_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        use_mcp: bool,
        has_pdf: bool,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
        }
        if has_pdf:
            payload.setdefault("metadata", {})["has_pdf"] = True
        if use_mcp:
            await self._inject_tools(payload)
        return payload

    async def stream_response(
        self,
        payload: Dict[str, Any],
        *,
        use_mcp: bool,
        mcp_auto_approve: bool,
        accumulated_tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        calls_buffer: List[Dict[str, Any]] = list(accumulated_tool_calls or [])

        if use_mcp and calls_buffer:
            payload = await self._execute_tools(calls_buffer, payload)
            calls_buffer.clear()

        try:
            headers = self._build_headers()
        except RuntimeError as exc:
            yield {"error": str(exc)}
            return

        follow_up_with_tools = False

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    self._chat_endpoint(),
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices", [])
                        for choice in choices:
                            delta = choice.get("delta", {})
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                self._accumulate_tool_calls(tool_calls, calls_buffer)
                            finish_reason = choice.get("finish_reason")
                            if (
                                use_mcp
                                and calls_buffer
                                and finish_reason
                                in {"tool_calls", "tool_call"}
                            ):
                                follow_up_with_tools = True
                        yield event
        except httpx.HTTPStatusError as exc:
            error_payload = {
                "error": f"Upstream error {exc.response.status_code}",
                "details": exc.response.text[:400],
            }
            yield error_payload
            return
        except Exception as exc:  # pragma: no cover - defensive
            yield {"error": str(exc)}
            return

        if not use_mcp:
            return

        if not calls_buffer:
            return

        if mcp_auto_approve or follow_up_with_tools:
            payload = await self._execute_tools(calls_buffer, payload)
            async for event in self.stream_response(
                payload,
                use_mcp=use_mcp,
                mcp_auto_approve=mcp_auto_approve,
            ):
                yield event
            return

        # Surface pending tool calls to the frontend so it can approve/deny.
        yield {
            "type": "tool_calls_pending",
            "tool_calls": calls_buffer,
        }

    async def _inject_tools(self, payload: Dict[str, Any]) -> None:
        try:
            clients = await mcp_manager.get_or_create_all_clients()
        except Exception as exc:
            payload.setdefault("metadata", {})["mcp_error"] = str(exc)
            payload["tools"] = []
            return

        tools_result = await asyncio.gather(
            *[client.get_available_tools() for client in clients],
            return_exceptions=True,
        )

        tools: List[Dict[str, Any]] = []
        for item in tools_result:
            if isinstance(item, Exception):
                continue
            tools.extend(item)
        if tools:
            payload["tools"] = tools

    def _accumulate_tool_calls(
        self, deltas: List[Dict[str, Any]], accumulator: List[Dict[str, Any]]
    ) -> None:
        for delta in deltas:
            call_id = delta.get("id")
            if not call_id:
                continue
            existing = next((c for c in accumulator if c.get("id") == call_id), None)
            if not existing:
                existing = {
                    "id": call_id,
                    "type": delta.get("type", "function"),
                    "function": {"name": "", "arguments": ""},
                }
                accumulator.append(existing)
            function_delta = delta.get("function", {})
            if function_delta.get("name"):
                existing["function"]["name"] = function_delta["name"]
            if function_delta.get("arguments"):
                existing["function"]["arguments"] = (
                    existing["function"].get("arguments", "")
                    + function_delta["arguments"]
                )

    async def _execute_tools(
        self, tool_calls: List[Dict[str, Any]], payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        for call in tool_calls:
            name = ((call.get("function") or {}).get("name") or "").strip()
            raw_args = (call.get("function") or {}).get("arguments", "")
            try:
                args = json.loads(raw_args) if raw_args else {}
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must decode to an object")
            except Exception as exc:
                args = {}
                result = {
                    "success": False,
                    "error": f"Failed to parse tool arguments: {exc}",
                }
            else:
                result = await mcp_manager.call_tool(name, args)
            payload.setdefault("messages", []).append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": json.dumps(result),
                }
            )
        return payload

    def _build_headers(self) -> Dict[str, str]:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_REFERER")
        if referer:
            headers["HTTP-Referer"] = referer
        title = os.getenv("OPENROUTER_APP")
        if title:
            headers["X-Title"] = title
        return headers

    def _chat_endpoint(self) -> str:
        endpoint = (
            (self.model_data.get("endpoints") or {}).get("openai")
            if isinstance(self.model_data, dict)
            else None
        )
        return endpoint or OPENROUTER_CHAT_URL
