import json

import tracking_pb2


def agent_result_to_chat_response(result, audio_bytes: bytes = b"") -> tracking_pb2.ChatResponse:
    payload_json = json.dumps(result.payload) if result.payload else "{}"
    return tracking_pb2.ChatResponse(
        response=result.reply_text,
        audio_response=audio_bytes,
        agent_name=result.agent_name,
        agent_state=result.state,
        agent_payload=payload_json,
    )
