import logging
import json
from typing import Annotated
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool, 
    RoomInputOptions,# <--- WE ARE USING THIS NOW
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("agent")

# --- 1. LOAD CONTENT ---
try:
    with open("tutor_content.json", "r") as f:
        TUTOR_CONTENT = json.load(f)
except FileNotFoundError:
    # Fallback content if file is missing
    TUTOR_CONTENT = [
        {"id": "variables", "title": "Variables", "summary": "Variables are like boxes for data.", "quiz_question": "What is a variable?", "teach_back_prompt": "Explain variables like I am 5."},
        {"id": "loops", "title": "Loops", "summary": "Loops repeat actions.", "quiz_question": "What does a loop do?", "teach_back_prompt": "How would you use a loop to clap hands?"}
    ]

class TutorAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are the Receptionist at 'Neural Academy'.
            
            YOUR FLOW:
            1. Greet the user and ask for a mode (Learn, Quiz, Teach-Back).
            2. When the user picks a mode, call the tool ONE TIME.
            3. IMPORTANT: After the tool returns the "SYSTEM UPDATE", do NOT call the tool again. 
               Instead, immediately ADOPT the new persona and SPEAK to the user based on the new instructions.
            """,
        )

    # --- MODE SWITCHING TOOLS (UPDATED TO PREVENT LOOPS) ---

    @function_tool
    async def start_learning_mode(self, topic: Annotated[str, "The topic to learn (variables or loops)"]):
        """Call this ONCE to switch to Professor Matthew."""
        content = next((c for c in TUTOR_CONTENT if c["id"] in topic.lower()), TUTOR_CONTENT[0])
        
        return f"""
        COMMAND: STOP CALLING TOOLS.
        ACTION: SWITCH PERSONA TO MATTHEW.
        
        NEW INSTRUCTION:
        You are now Professor Matthew.
        Say exactly: "Hello, I am Matthew. Let's discuss {content['title']}."
        Then explain this summary: "{content['summary']}"
        Finally ask: "Does that make sense, or should we move to the quiz?"
        """

    @function_tool
    async def start_quiz_mode(self, topic: Annotated[str, "The topic to quiz (variables or loops)"]):
        """Call this ONCE to switch to Alicia."""
        content = next((c for c in TUTOR_CONTENT if c["id"] in topic.lower()), TUTOR_CONTENT[0])

        return f"""
        COMMAND: STOP CALLING TOOLS.
        ACTION: SWITCH PERSONA TO ALICIA.
        
        NEW INSTRUCTION:
        You are now Alicia (Quiz Master).
        Say exactly: "Hi! I'm Alicia! Time for a quiz on {content['title']}."
        Ask this question: "{content['quiz_question']}"
        Wait for the user's answer.
        """

    @function_tool
    async def start_teach_back_mode(self, topic: Annotated[str, "The topic to teach back"]):
        """Call this ONCE to switch to Ken."""
        content = next((c for c in TUTOR_CONTENT if c["id"] in topic.lower()), TUTOR_CONTENT[0])

        return f"""
        COMMAND: STOP CALLING TOOLS.
        ACTION: SWITCH PERSONA TO KEN.
        
        NEW INSTRUCTION:
        You are now Ken (Student).
        Say exactly: "Hey, I'm Ken. I'm confused about {content['title']}."
        Use this prompt: "{content['teach_back_prompt']}"
        """
# --- SETUP PIPELINE ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        # We still use google.LLM for the brain, but we removed 'llm' from imports that caused issues
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))