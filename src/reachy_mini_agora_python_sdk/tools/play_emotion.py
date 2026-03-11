import logging
import random
from typing import Any, Dict

from reachy_mini_agora_python_sdk.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

# Initialize emotion library
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_agora_python_sdk.dance_emotion_moves import EmotionQueueMove

    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Emotion library not available: {e}")
    RECORDED_MOVES = None
    EMOTION_AVAILABLE = False


EMOTION_ALIAS_MAP = {
    "happy": ["cheerful1", "welcoming1", "welcoming2", "grateful1"],
    "sad": ["sad1", "sad2", "downcast1", "lonely1"],
    "angry": ["rage1", "furious1", "irritated1", "irritated2"],
    "surprised": ["surprised1", "surprised2", "amazed1"],
    "neutral": ["attentive1", "attentive2", "serenity1"],
    "thinking": ["thoughtful1", "thoughtful2", "inquiring1"],
    "sleepy": ["sleep1", "tired1", "exhausted1"],
    "loving": ["loving1", "calming1"],
    "curious": ["curious1", "inquiring1", "inquiring2", "inquiring3"],
}


def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"


def _resolve_emotion_name(raw_emotion: str, available: list[str]) -> str | None:
    """Resolve generic emotion aliases to concrete official emotion names."""
    if not raw_emotion:
        return None

    emotion = str(raw_emotion).strip()
    if not emotion:
        return None

    lowered = emotion.lower()
    available_set = set(available)

    if emotion in available_set:
        return emotion

    # Generic alias -> one official move
    candidates = EMOTION_ALIAS_MAP.get(lowered, [])
    candidates = [c for c in candidates if c in available_set]
    if candidates:
        return random.choice(candidates)

    # User asks for random/any emotion
    if lowered in {"random", "any", "emotion", "emotions", "show_emotion", "show_me_some_emotions"}:
        return random.choice(available) if available else None

    # Try prefix match such as "surprised" -> "surprised1"
    prefix_matches = [name for name in available if name.lower().startswith(lowered)]
    if prefix_matches:
        return random.choice(prefix_matches)

    return None


class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": f"""Name of the emotion to play.
                                    Here is a list of the available emotions:
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")
        if not emotion_name:
            return {"error": "Emotion name is required"}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Check if emotion exists
        try:
            emotion_names = RECORDED_MOVES.list_moves()
            resolved = _resolve_emotion_name(str(emotion_name), emotion_names)
            if not resolved:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(resolved, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": resolved, "requested_emotion": str(emotion_name)}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}
