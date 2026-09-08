# AI_OWNED
"""Talk to the agent (docs/VOICE.md).

	voice.turns    turn-taking decisions: endpointing, barge-in, what to say while waiting; pure
	voice.vad      Silero VAD v5 through onnxruntime, one probability per 32 ms frame
	voice.speech   the STT and TTS calls against the OpenAI-shaped gateway
	voice.client   the host-side program: microphone, speaker, and the agent subprocess
	voice.bridge   the container-side program: agent_loop as a JSON-lines protocol
"""
