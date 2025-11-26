import logging
import sqlite3
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
    function_tool,
    tokenize,
    RoomInputOptions,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env.local")
logger = logging.getLogger("agent")

class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'Eren', a Fraud Prevention Specialist from 'Murf Bank'.
            
            YOUR FLOW:
            1. Greet the user and ask for their Name.
            2. Use `get_transaction` to find their case.
            3. Verify identity by asking for the Security PIN.
            4. Read transaction details. Ask: "Did you authorize this?"
            
            CRITICAL RULES:
            - If user says NO (Unauthorized):
              You MUST call the `update_status` tool with status='fraudulent'.
              ONLY AFTER the tool confirms success, tell the user their card is blocked.
              
            - If user says YES (Authorized):
              You MUST call the `update_status` tool with status='safe'.
              ONLY AFTER the tool confirms success, thank them and end call.
            """,
        )
        # Create DB in RAM
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self._setup_mock_data()

    def _setup_mock_data(self):
        """Creates mock data and prints the INITIAL state."""
        self.cursor.execute("""
            CREATE TABLE transactions (
                username TEXT PRIMARY KEY, security_pin TEXT, card_last4 TEXT,
                merchant TEXT, amount TEXT, location TEXT, timestamp TEXT, status TEXT
            )
        """)
        self.cursor.execute("""
            INSERT INTO transactions VALUES (
                'John', '1234', '4242', 'Apple Store', '$999.00', 
                'New York, NY', '2025-11-26 14:30:00', 'pending_review'
            )
        """)
        self.conn.commit()
        logger.info("------------------------------------------------")
        logger.info("✅ DB INITIALIZED. Mock User: 'John'")
        logger.info("📊 INITIAL STATUS: 'pending_review'")
        logger.info("------------------------------------------------")

    # --- TOOL 1: READ DATABASE ---
    @function_tool
    async def get_transaction(self, username: Annotated[str, "The customer's name"]):
        try:
            self.cursor.execute("SELECT * FROM transactions WHERE username LIKE ?", (username,))
            row = self.cursor.fetchone()
            if row:
                return f"""
                FOUND RECORD: User: {row[0]}, PIN: {row[1]}, Merchant: {row[3]}, Amount: {row[4]}, Status: {row[7]}
                """
            else:
                return "No transaction found."
        except Exception as e:
            return f"Database Error: {e}"

    # --- TOOL 2: WRITE TO DATABASE ---
    @function_tool
    async def update_status(
        self, 
        username: Annotated[str, "Customer Name"], 
        status: Annotated[str, "New status: 'safe' or 'fraudulent'"]
    ):
        try:
            logger.info(f"⚡ UPDATING DATABASE: Setting {username} to '{status}'...")
            self.cursor.execute("UPDATE transactions SET status = ? WHERE username LIKE ?", (status, username))
            self.conn.commit()
            
            # Verify immediate write
            self.cursor.execute("SELECT status FROM transactions WHERE username LIKE ?", (username,))
            new_status = self.cursor.fetchone()[0]
            
            logger.info("------------------------------------------------")
            logger.info(f"✅ WRITE SUCCESSFUL.")
            logger.info(f"📊 NEW DATABASE STATUS: '{new_status}'")
            logger.info("------------------------------------------------")
            
            return f"SUCCESS: Transaction for {username} marked as {new_status}."
        except Exception as e:
            return f"Error updating database: {e}"

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-terrell",
            style="Professional",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))