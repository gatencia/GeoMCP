"""Utility classes for Model Context Protocol client management.

This module provides a thin wrapper around the FastMCP client so that the
GeoMCP backend can discover tools, execute tool calls, and share connections
across chat sessions.  The implementation follows the integration plan used
by the streaming chat endpoint and keeps a small cache of connected clients.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # FastMCP is optional during cold starts
    from fastmcp import Client
except ImportError:  # pragma: no cover - handled at runtime
    Client = None  # type: ignore


class MCPClient:
    """Async wrapper for a single MCP server connection."""

    client: Optional[Any]
    available_tools: List[Dict[str, Any]]
    connected: bool
    _config: Optional[Dict[str, Any]]

    def __init__(self) -> None:
        self.client = None
        self.available_tools = []
        self.connected = False
        self._config = None
        self._lock = asyncio.Lock()

    def convert_tool_format(self, tool: Any) -> Dict[str, Any]:
        """Convert the tool definition into an OpenAI-compatible structure."""
        return {
            "type": "function",
            "function": {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", "") or "",
                "parameters": getattr(tool, "inputSchema", None)
                or {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }

    async def connect_to_server(self, server_config: Dict[str, Any]) -> bool:
        """Connect to a server using the supplied configuration."""
        if Client is None:
            print("FastMCP not available")
            return False

        async with self._lock:
            # Cache the config so call_tool can recreate clients when needed
            self._config = server_config
            try:
                client = Client(self._build_client_options(server_config))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"Failed to initialise MCP client: {exc}")
                self.connected = False
                return False

            try:
                async with asyncio.timeout(10.0):
                    async with client:
                        await client.ping()
                        tools = await client.list_tools()
                        self.available_tools = [
                            self.convert_tool_format(tool) for tool in tools
                        ]
                        self.client = client
                        self.connected = True
                        tool_names = [
                            tool["function"]["name"]
                            for tool in self.available_tools
                        ]
                        print(
                            "Connected to MCP server with tools:",
                            ", ".join(tool_names) if tool_names else "<none>",
                        )
                        return True
            except asyncio.TimeoutError:
                print("Timeout connecting to MCP server")
            except Exception as exc:
                print(f"Failed to connect to MCP server: {exc}")

            self.connected = False
            self.client = None
            return False

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return tool metadata if connected."""
        if not self.connected:
            return []
        return self.available_tools

    async def call_tool(
        self, tool_name: str, tool_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool through the configured MCP client."""
        print(f"Calling tool {tool_name} with args {tool_args}")
        if Client is None or not self._config:
            return {
                "success": False,
                "error": "FastMCP client not available",
                "tool_name": tool_name,
                "tool_args": tool_args,
            }

        async with self._lock:
            client = Client(self._build_client_options(self._config))
            try:
                async with asyncio.timeout(30.0):
                    async with client:
                        result = await client.call_tool(tool_name, tool_args)
                        print(f"Tool {tool_name} executed successfully")
                        content: List[str] = []
                        if hasattr(result, "content") and result.content:
                            for item in result.content:
                                if hasattr(item, "text"):
                                    content.append(item.text)  # type: ignore[attr-defined]
                                elif hasattr(item, "data"):
                                    content.append(str(item.data))
                                else:
                                    content.append(str(item))
                        return {
                            "success": True,
                            "content": content,
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                        }
            except asyncio.TimeoutError:
                error_msg = f"Tool {tool_name} timed out after 30 seconds"
                print(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }
            except Exception as exc:  # pragma: no cover - defensive
                error_msg = f"Error executing tool {tool_name}: {exc}"
                print(error_msg)
                return {
                    "success": False,
                    "error": str(exc),
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                }

    async def cleanup(self) -> None:
        """Reset the client state."""
        self.connected = False
        self.client = None
        self.available_tools = []
        self._config = None

    def _build_client_options(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise configuration for the FastMCP client."""
        if config.get("command") and config.get("args"):
            return {
                "mcpServers": {
                    "default": {
                        "transport": "stdio",
                        "command": config["command"],
                        "args": config.get("args", []),
                        "env": config.get("env"),
                    }
                }
            }
        if config.get("url"):
            return {
                "mcpServers": {
                    "default": {
                        "transport": "http",
                        "url": config["url"],
                    }
                }
            }
        raise ValueError("Invalid server config - no command or URL provided")


class MCPManager:
    """Store and reuse MCP clients based on configuration keys."""

    def __init__(self) -> None:
        self.clients: Dict[str, MCPClient] = {}
        self.default_configs = self._load_default_configs()

    async def get_or_create_client(
        self, server_type: str = "geomcp", custom_config: Optional[Dict[str, Any]] = None
    ) -> MCPClient:
        client_key = f"{server_type}_{hash(str(custom_config) if custom_config else '')}"
        if client_key not in self.clients:
            client = MCPClient()
            config = custom_config or self.default_configs.get(server_type)
            if not config:
                raise ValueError(f"No configuration found for server type: {server_type}")
            success = await client.connect_to_server(config)
            if not success:
                raise RuntimeError(f"Failed to connect to {server_type} MCP server")
            self.clients[client_key] = client
        return self.clients[client_key]

    async def get_or_create_all_clients(self) -> List[MCPClient]:
        clients: List[MCPClient] = []
        for server_type in self.default_configs.keys():
            try:
                print(f"Creating MCP client for server type: {server_type}")
                client = await self.get_or_create_client(server_type)
                clients.append(client)
            except Exception as exc:
                print(f"Error creating client for {server_type}: {exc}")
        return clients

    async def call_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        for client in self.clients.values():
            if any(tool["function"]["name"] == tool_name for tool in client.available_tools):
                return await client.call_tool(tool_name, tool_args)
        return {
            "success": False,
            "error": f"No connected MCP client has tool {tool_name}",
            "tool_name": tool_name,
            "tool_args": tool_args,
        }

    async def cleanup_all(self) -> None:
        for client in self.clients.values():
            await client.cleanup()
        self.clients.clear()

    def _load_default_configs(self) -> Dict[str, Dict[str, Any]]:
        config_path = Path(__file__).with_name("mcp_servers.json")
        if not config_path.exists():
            return {}
        try:
            with config_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed to load mcp_servers.json: {exc}")
            return {}


# Global manager instance reused across requests
mcp_manager = MCPManager()
