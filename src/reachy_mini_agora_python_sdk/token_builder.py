"""Agora RTC Token Builder.

This module generates RTC tokens for Agora channels.
Based on: https://github.com/AgoraIO/Tools/blob/master/DynamicKey/AgoraDynamicKey/python3/sample/RtcTokenBuilder2Sample.py
"""

import logging
import time

try:
    from agora_token_builder import RtcTokenBuilder
except ImportError:
    RtcTokenBuilder = None


class TokenGenerator:
    """Generator for Agora RTC tokens."""

    def __init__(self, app_id: str, app_certificate: str):
        """Initialize the token generator.

        Args:
            app_id: Agora application ID
            app_certificate: Agora application certificate (empty string if not enabled)
        """
        self.app_id = app_id
        self.app_certificate = app_certificate
        self.logger = logging.getLogger(__name__)

        # Check if certificate is enabled
        self.certificate_enabled = bool(app_certificate and app_certificate.strip())

        if self.certificate_enabled:
            if RtcTokenBuilder is None:
                self.logger.warning(
                    "agora-token-builder not installed. "
                    "Install it with: pip install agora-token-builder"
                )
            else:
                self.logger.info("Token generation enabled (certificate is set)")
        else:
            self.logger.info("Token generation disabled (no certificate)")

    def generate_rtc_token(
        self,
        channel_name: str,
        uid: int,
        role: int = 1,
        expire_time: int = 3600
    ) -> str:
        """Generate RTC token for a user.

        Args:
            channel_name: Channel name
            uid: User ID (integer)
            role: User role (1=publisher, 2=subscriber)
            expire_time: Token expiration time in seconds (default: 3600 = 1 hour)

        Returns:
            Token string (empty if certificate not enabled)
        """
        # If certificate not enabled, return empty token
        if not self.certificate_enabled:
            self.logger.debug("Certificate not enabled, returning empty token")
            return ""

        # Check if token builder is available
        if RtcTokenBuilder is None:
            self.logger.error(
                "Cannot generate token: agora-token-builder not installed"
            )
            return ""

        try:
            # Calculate privilege expiration timestamp
            current_timestamp = int(time.time())
            privilege_expired_ts = current_timestamp + expire_time

            # Build token using the correct method name
            token = RtcTokenBuilder().buildTokenWithUid(
                self.app_id,
                self.app_certificate,
                channel_name,
                uid,
                role,
                privilege_expired_ts
            )

            self.logger.info(
                f"Generated token for channel={channel_name}, "
                f"uid={uid}, expires_in={expire_time}s"
            )

            return token

        except Exception as e:
            self.logger.error(f"Error generating token: {e}")
            return ""

    def generate_token_for_user(
        self,
        channel_name: str,
        uid: int,
        expire_time: int = 3600
    ) -> str:
        """Generate token for a user (publisher role).

        Args:
            channel_name: Channel name
            uid: User ID
            expire_time: Token expiration time in seconds

        Returns:
            Token string
        """
        return self.generate_rtc_token(
            channel_name=channel_name,
            uid=uid,
            role=1,  # Publisher
            expire_time=expire_time
        )

    def generate_token_for_agent(
        self,
        channel_name: str,
        agent_uid: int = 0,
        expire_time: int = 3600
    ) -> str:
        """Generate token for AI agent.

        Args:
            channel_name: Channel name
            agent_uid: Agent UID (0 for auto-assign)
            expire_time: Token expiration time in seconds

        Returns:
            Token string
        """
        return self.generate_rtc_token(
            channel_name=channel_name,
            uid=agent_uid,
            role=1,  # Publisher
            expire_time=expire_time
        )

    def is_certificate_enabled(self) -> bool:
        """Check if certificate is enabled.

        Returns:
            True if certificate is enabled, False otherwise
        """
        return self.certificate_enabled


# Fallback implementation if agora-token-builder is not available
class RtcTokenBuilderFallback:
    """Fallback token builder using manual implementation."""

    @staticmethod
    def build_token_with_uid(
        app_id: str,
        app_certificate: str,
        channel_name: str,
        uid: int,
        role: int,
        token_expire: int,
        privilege_expire: int
    ) -> str:
        """Build token manually (simplified version).

        Note: This is a placeholder. For production, use the official
        agora-token-builder package.
        """
        raise NotImplementedError(
            "Token generation requires agora-token-builder package. "
            "Install it with: pip install agora-token-builder"
        )
