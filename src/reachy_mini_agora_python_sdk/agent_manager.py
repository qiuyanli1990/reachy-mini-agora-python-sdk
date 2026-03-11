"""Agora Conversational AI Agent Manager (Version 2).

This version supports loading agent configuration from JSON file.
"""

import logging
import base64
import copy
import json
import time
import re
from typing import Optional, Any
from pathlib import Path

import requests


class AgentManager:
    """Manager for Agora Conversational AI Agent.

    This class handles the REST API calls to start and stop
    conversational AI agents in Agora channels.
    Supports loading configuration from JSON file.
    """

    def __init__(self, app_id: str, api_key: str, api_secret: str, config_file: str = "agent_config.json"):
        """Initialize the agent manager.

        Args:
            app_id: Agora application ID
            api_key: API key for authentication
            api_secret: API secret for authentication
            config_file: Path to agent configuration JSON file
        """
        self.app_id = app_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.config_file = Path(config_file)
        self.agent_id: Optional[str] = None
        self.agent_config: Optional[dict[str, Any]] = None

        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # API endpoints
        self.base_url = "https://api.agora.io/api/conversational-ai-agent/v2"

        # Load config if file exists
        if self.config_file.exists():
            self.load_config()
        else:
            self.logger.warning(f"Config file not found: {config_file}")

        self.logger.info("AgentManager initialized")

    def load_config(self) -> bool:
        """Load agent configuration from JSON file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.logger.info(f"Loading agent config from: {self.config_file}")

            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.agent_config = json.load(f)

            # Render placeholder templates like "{{prompt.txt}}".
            self._render_prompt_placeholders()

            self.logger.info("Agent config loaded successfully")
            return True

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            return False

    def _render_prompt_placeholders(self) -> None:
        """Render placeholders like {{prompt.txt}} in the loaded JSON config.

        Resolution is relative to the directory of `agent_config.json`.
        """
        if not isinstance(self.agent_config, dict):
            return

        pattern = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
        base_dir = self.config_file.parent

        def resolve_placeholder(file_expr: str) -> str:
            candidate = Path(file_expr.strip())
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.warning("Failed to read placeholder file %s: %s", candidate, e)
                return ""

        def replace_in_string(s: str) -> str:
            def repl(match: re.Match[str]) -> str:
                expr = match.group(1)
                return resolve_placeholder(expr)
            return pattern.sub(repl, s)

        def walk(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: walk(v) for k, v in node.items()}
            if isinstance(node, list):
                return [walk(v) for v in node]
            if isinstance(node, str) and "{{" in node and "}}" in node:
                return replace_in_string(node)
            return node

        self.agent_config = walk(self.agent_config)

    def _get_auth_header(self) -> str:
        """Generate Basic Auth header.

        Returns:
            Base64 encoded authentication string
        """
        credentials = f"{self.api_key}:{self.api_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def start_agent_from_config(self, channel_name: str, user_uid: int, token: str = "") -> bool:
        """Start agent using configuration from JSON file.

        Args:
            channel_name: Channel name to join
            user_uid: User UID to subscribe to
            token: Optional RTC token

        Returns:
            True if agent started successfully, False otherwise
        """
        if not self.agent_config:
            self.logger.error("No agent config loaded. Call load_config() first.")
            return False

        # Update channel info
        self.agent_config["properties"]["channel"] = channel_name
        self.agent_config["properties"]["token"] = token
        
        # Update remote UIDs to subscribe to specific user
        if user_uid:
            self.agent_config["properties"]["remote_rtc_uids"] = [str(user_uid)]

        return self.start_agent_with_payload(self.agent_config)

    def start_agent_with_payload(self, payload: dict[str, Any]) -> bool:
        """Start agent with custom payload.

        Args:
            payload: Complete agent configuration payload

        Returns:
            True if agent started successfully, False otherwise
        """
        try:
            self.logger.info("Starting agent with custom payload...")
            normalized_payload = self._normalize_payload_for_api(payload)

            # Build API URL
            url = f"{self.base_url}/projects/{self.app_id}/join"

            # Build headers
            headers = {
                "Authorization": self._get_auth_header(),
                "Content-Type": "application/json"
            }

            # Log payload (without sensitive info)
            self._log_start_payloads(payload, normalized_payload)

            # Make API request
            response = requests.post(url, headers=headers, json=normalized_payload, timeout=30)

            # Check response
            if response.status_code == 200:
                result = response.json()
                self.agent_id = result.get("agent_id") or result.get("agentId")
                self.logger.info(f"✓ Agent started successfully: {self.agent_id}")
                return True

            # Handle task conflict by stopping running agent then retrying once
            if response.status_code == 409:
                conflict_agent_id = self._extract_conflict_agent_id(response)
                self.logger.warning(
                    "Start agent conflict (409). Trying stop-then-retry flow..."
                )
                if self._handle_task_conflict_and_retry(
                    url, headers, normalized_payload, conflict_agent_id
                ):
                    return True

            self.logger.error(
                f"Failed to start agent: {response.status_code} - {response.text}"
            )
            return False

        except requests.exceptions.Timeout:
            self.logger.error("Request timeout - agent may still be starting")
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error starting agent: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error starting agent: {e}")
            return False

    def _normalize_payload_for_api(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Normalize payload to avoid common ConvoAI schema validation errors."""
        normalized = copy.deepcopy(payload)
        props = normalized.get("properties", {})
        if not isinstance(props, dict):
            return normalized

        # Ensure tool bridge path is always enabled for local tool dispatch.
        advanced = props.get("advanced_features")
        if not isinstance(advanced, dict):
            advanced = {}
        advanced["enable_tools"] = True
        props["advanced_features"] = advanced

        parameters = props.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        parameters["data_channel"] = "datastream"
        props["parameters"] = parameters

        llm = props.get("llm", {})
        if isinstance(llm, dict):
            # Local helper field only; never send to ConvoAI API.
            llm.pop("system_prompt_file", None)

            # greeting_configs.mode must be valid if present.
            greeting_cfg = llm.get("greeting_configs")
            if isinstance(greeting_cfg, dict):
                mode = greeting_cfg.get("mode")
                allowed = {"single_every", "single_first"}
                if not isinstance(mode, str) or not mode.strip() or mode not in allowed:
                    greeting_cfg.pop("mode", None)
                if not greeting_cfg:
                    llm.pop("greeting_configs", None)
            elif greeting_cfg is not None:
                llm.pop("greeting_configs", None)

            # ConvoAI expects a list.
            if "predefined_tools" in llm:
                tools = llm.get("predefined_tools")
                if isinstance(tools, str):
                    tools_list = [t.strip() for t in tools.split(",") if t.strip()]
                elif tools is None:
                    tools_list = []
                elif not isinstance(tools, list):
                    tools_list = [str(tools)]
                else:
                    tools_list = [str(t).strip() for t in tools if str(t).strip()]
            else:
                tools_list = []

            # Keep only ConvoAI-supported bridge tool.
            filtered_out = [t for t in tools_list if t != "_publish_message"]
            if filtered_out:
                self.logger.info(
                    "Ignoring non-ConvoAI predefined_tools entries: %s. "
                    "Local actions are dispatched via _publish_message content.",
                    filtered_out,
                )
            llm["predefined_tools"] = ["_publish_message"]

            # Normalize system message schemas (Gemini-like `parts` -> `content`).
            llm["system_messages"] = self._normalize_system_messages(
                llm.get("system_messages", [])
            )

            props["llm"] = llm

        normalized["properties"] = props
        return normalized

    def _normalize_system_messages(self, messages: Any) -> list[dict[str, str]]:
        """Normalize system message list into OpenAI-like role/content format."""
        if not isinstance(messages, list):
            return []

        normalized: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = str(msg.get("role", "system")).strip() or "system"
            content = msg.get("content")

            if isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content})
                continue

            parts = msg.get("parts")
            if isinstance(parts, list):
                chunks: list[str] = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    txt = part.get("text")
                    if isinstance(txt, str) and txt:
                        chunks.append(txt)
                merged = "\n".join(chunks).strip()
                if merged:
                    normalized.append({"role": role, "content": merged})

        return normalized

    def _handle_task_conflict_and_retry(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        conflict_agent_id: Optional[str] = None,
    ) -> bool:
        """Handle 409 conflict by stopping current running agent and retrying start."""
        try:
            # Query running agent(s)
            running_agent_id = conflict_agent_id or self._query_running_agent_id(headers)
            if running_agent_id:
                self.logger.info(
                    "Found conflicting/running agent: %s. Stopping it first...",
                    running_agent_id,
                )
                if not self.stop_agent_by_id(running_agent_id):
                    self.logger.error(
                        "Failed to stop running agent %s during conflict handling",
                        running_agent_id,
                    )
                    return False
                # Give backend a short settle window to release task locks.
                time.sleep(1.2)
            else:
                self.logger.warning(
                    "No running agent found via list API during conflict handling"
                )
                # Even if list returns empty, conflict lock may still be releasing.
                time.sleep(1.2)

            # Retry with short backoff; TaskConflict may need a few seconds to clear.
            retry_delays = [0.8, 1.5, 2.5]
            for idx, delay_s in enumerate(retry_delays, start=1):
                self.logger.info(
                    "Retrying start agent after conflict recovery (attempt %s/%s)...",
                    idx,
                    len(retry_delays),
                )
                retry_response = requests.post(url, headers=headers, json=payload, timeout=30)
                if retry_response.status_code == 200:
                    result = retry_response.json()
                    self.agent_id = result.get("agent_id") or result.get("agentId")
                    self.logger.info(
                        "✓ Agent started successfully after retry: %s",
                        self.agent_id,
                    )
                    return True

                # If not conflict anymore, fail fast with server message.
                if retry_response.status_code != 409:
                    self.logger.error(
                        "Retry start failed (non-409): %s - %s",
                        retry_response.status_code,
                        retry_response.text,
                    )
                    return False

                self.logger.warning(
                    "Retry start still conflicted: %s - %s",
                    retry_response.status_code,
                    retry_response.text,
                )
                time.sleep(delay_s)

            self.logger.error("Retry start failed after all conflict-retry attempts")
            return False
        except Exception as e:
            self.logger.error(f"Error during 409 conflict handling: {e}")
            return False

    def _extract_conflict_agent_id(self, response: requests.Response) -> Optional[str]:
        """Extract conflicting agent_id from 409 response payload if present."""
        try:
            data = response.json()
            if isinstance(data, dict):
                agent_id = data.get("agent_id") or data.get("agentId")
                if isinstance(agent_id, str) and agent_id:
                    return agent_id
        except Exception:
            return None
        return None

    def _query_running_agent_id(self, headers: dict[str, str]) -> Optional[str]:
        """Query conversational AI API for running agent in current project."""
        try:
            url = f"{self.base_url}/projects/{self.app_id}/agents"
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                self.logger.warning(
                    "List agents failed: %s - %s",
                    response.status_code,
                    response.text,
                )
                return None

            data = response.json()
            # Defensive parsing for possible response shapes.
            candidates = []
            if isinstance(data, dict):
                if isinstance(data.get("agents"), list):
                    candidates = data.get("agents", [])
                elif isinstance(data.get("data"), list):
                    candidates = data.get("data", [])
                elif isinstance(data.get("items"), list):
                    candidates = data.get("items", [])

            for item in candidates:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status", "")).lower()
                if status in {"running", "started", "active"}:
                    return item.get("agent_id") or item.get("agentId")

            # Fallback: if list exists but no clear status, take first with an id.
            for item in candidates:
                if not isinstance(item, dict):
                    continue
                candidate_id = item.get("agent_id") or item.get("agentId")
                if candidate_id:
                    return candidate_id

            return None
        except Exception as e:
            self.logger.warning(f"Query running agent failed: {e}")
            return None

    def stop_agent_by_id(self, agent_id: str) -> bool:
        """Stop a specific agent by id."""
        if not agent_id:
            return False

        try:
            headers = {
                "Authorization": self._get_auth_header(),
                "Content-Type": "application/json"
            }

            # Prefer v2 leave endpoint, keep legacy stop endpoint as fallback.
            stop_candidates = [
                (f"{self.base_url}/projects/{self.app_id}/agents/{agent_id}/leave", None),
                (f"{self.base_url}/projects/{self.app_id}/agents/{agent_id}/stop", None),
                # Some integrations use project-level leave with body.
                (f"{self.base_url}/projects/{self.app_id}/leave", {"agent_id": agent_id}),
            ]

            last_status = None
            last_text = ""
            for url, payload in stop_candidates:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                last_status = response.status_code
                last_text = response.text

                if response.status_code in (200, 204):
                    self.logger.info("✓ Agent stopped successfully: %s", agent_id)
                    if self.agent_id == agent_id:
                        self.agent_id = None
                    return True

                if response.status_code == 404:
                    # Treat as idempotent success: agent likely already stopped/removed.
                    self.logger.warning(
                        "Stop endpoint returned 404 for %s (url=%s), assuming already stopped",
                        agent_id,
                        url,
                    )
                    if self.agent_id == agent_id:
                        self.agent_id = None
                    return True

            self.logger.error(
                "Failed to stop agent %s after trying all endpoints: %s - %s",
                agent_id,
                last_status,
                last_text,
            )
            return False
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error stopping agent {agent_id}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error stopping agent {agent_id}: {e}")
            return False

    def _sanitize_payload_for_logging(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Remove sensitive information from payload for logging.

        Args:
            payload: Original payload

        Returns:
            Sanitized payload
        """
        safe = copy.deepcopy(payload)
        
        # Mask API keys
        if "properties" in safe:
            props = safe["properties"]
            if "llm" in props and "api_key" in props["llm"]:
                props["llm"]["api_key"] = "***MASKED***"
            if "asr" in props and "params" in props["asr"] and "api_key" in props["asr"]["params"]:
                props["asr"]["params"]["api_key"] = "***MASKED***"
            if "tts" in props and "params" in props["tts"] and "key" in props["tts"]["params"]:
                props["tts"]["params"]["key"] = "***MASKED***"
        
        return safe

    def _log_start_payloads(self, raw_payload: dict[str, Any], normalized_payload: dict[str, Any]) -> None:
        """Log sanitized payloads used for start request debugging."""
        safe_raw = self._sanitize_payload_for_logging(raw_payload)
        safe_normalized = self._sanitize_payload_for_logging(normalized_payload)
        self.logger.info(
            "Agent start body before normalization (sanitized): %s",
            json.dumps(safe_raw, ensure_ascii=False, indent=2),
        )
        self.logger.info(
            "Agent start body after normalization (sanitized): %s",
            json.dumps(safe_normalized, ensure_ascii=False, indent=2),
        )

    def stop_agent(self) -> bool:
        """Stop the currently running agent.

        Returns:
            True if agent stopped successfully, False otherwise
        """
        if not self.agent_id:
            self.logger.warning("No agent to stop")
            return False

        self.logger.info(f"Stopping agent: {self.agent_id}")
        return self.stop_agent_by_id(self.agent_id)

    def is_agent_running(self) -> bool:
        """Check if an agent is currently running.

        Returns:
            True if agent is running, False otherwise
        """
        return self.agent_id is not None
