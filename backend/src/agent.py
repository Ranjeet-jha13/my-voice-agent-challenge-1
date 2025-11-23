import logging
import json
import os
from datetime import datetime
from typing import Annotated
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        # 1. READ MEMORY: Check if we have previous data to reference
        past_context = ""
        file_path = "wellness.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    if data and len(data) > 0:
                        last_entry = data[-1] # Get the most recent entry
                        past_context = (
                            f"CONTEXT FROM LAST SESSION: "
                            f"On {last_entry.get('timestamp', 'unknown date')}, "
                            f"the user felt '{last_entry.get('mood', 'unknown')}' "
                            f"with energy level '{last_entry.get('energy', 'unknown')}'. "
                            f"Their goal was: '{last_entry.get('objectives', 'unknown')}'."
                            f"Start the conversation by briefly mentioning this."
                        )
            except Exception:
                past_context = "No previous records found. This is your first session."

        super().__init__(
            instructions=f"""You are Coach Alex, a supportive and empathetic Health & Wellness Companion.
            Your goal is to perform a Daily Check-in with the user.
            
            {past_context}

            FOLLOW THIS SCRIPT FLOW:
            1. **Mood & Energy:** Ask "How are you feeling today?" and "What is your energy like?".
            2. **Intentions:** Ask "What are 1-3 things you want to get done today?".
            3. **Advice:** Offer simple, realistic advice (e.g., take a walk, drink water).
            4. **Recap:** Summarize their mood and goals back to them.
            5. **SAVE:** Once they confirm the recap, call the 'log_daily_checkin' tool.

            CAPABILITIES:
            - If they ask about weight/height, use the 'calculate_bmi' tool.
            - Do NOT diagnose medical conditions.
            - Keep responses warm but concise.
            """,
        )

    @function_tool
    async def calculate_bmi(
        self,
        weight_kg: Annotated[float, "The user's weight in kilograms"],
        height_cm: Annotated[float, "The user's height in centimeters"],
    ):
        """Calculate Body Mass Index (BMI) from height and weight."""
        height_m = height_cm / 100
        bmi = weight_kg / (height_m ** 2)
        bmi = round(bmi, 1)
        
        category = ""
        if bmi < 18.5: category = "underweight"
        elif bmi < 25: category = "healthy weight"
        elif bmi < 30: category = "overweight"
        else: category = "obese"

        return f"The BMI is {bmi}, which is considered {category}."

    @function_tool
    async def log_daily_checkin(
        self,
        mood: Annotated[str, "Summary of the user's mood"],
        energy: Annotated[str, "User's energy level"],
        objectives: Annotated[str, "The user's main goals for the day"],
    ):
        """Save the daily check-in summary (mood, energy, objectives) to a JSON file."""
        file_path = "wellness.json"
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = {
            "timestamp": timestamp,
            "type": "daily_checkin",
            "mood": mood,
            "energy": energy,
            "objectives": objectives
        }
        
        logger.info(f"Saving Check-in: {new_entry}")

        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except Exception:
                data = []

        data.append(new_entry)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return "Check-in saved successfully! Wishing you a great day ahead."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))