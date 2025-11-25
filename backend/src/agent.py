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
    WorkerOptions,
    cli,
    function_tool,
    llm, # Ensure this is imported for type hints if needed, but we avoid direct usage in events
    tokenize,
    RoomInputOptions,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from company_data import COMPANY_NAME, FAQ_DATA

load_dotenv(".env.local")
logger = logging.getLogger("agent")

class SDRAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions=f"""You are Samson, an SDR for {COMPANY_NAME}.
            
            YOUR GOAL:
            Qualify the user as a potential restaurant partner.
            
            KNOWLEDGE BASE:
            {FAQ_DATA}
            
            CONVERSATION FLOW:
            1. Greet them warmly and ask what brought them here.
            2. Answer their questions using the FAQ. Do not make up facts.
            3. SMOOTHLY ask for their details: Name, Restaurant Name, Email, and Timeline.
            4. Use the 'save_lead_details' tool as soon as you get this info.
            5. Once you have the info and answered questions, politely end the call.
            
            TONE: Professional, helpful, and concise.
            """,
        )
        # Initialize the transcript list
        self.transcript = []

    @function_tool
    async def save_lead_details(
        self, 
        name: Annotated[str, "Customer Name"],
        restaurant_name: Annotated[str, "Restaurant or Company Name"],
        email: Annotated[str, "Email Address"],
        timeline: Annotated[str, "When they want to start (Now/Soon/Later)"]
    ):
        """Saves the lead's core contact details to a JSON file."""
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "restaurant": restaurant_name,
            "email": email,
            "timeline": timeline
        }
        
        file_path = "leads.json"
        data = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
            except: pass
            
        data.append(entry)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
        return "Details saved. Thank you!"

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    sdr_agent = SDRAgent()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Professional",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
    )

    # --- THE FIX: USE 'conversation_item_added' ---
    # This event comes from the documentation you shared. 
    # It fires whenever a text message is added to the chat history.
    
    @session.on("conversation_item_added")
    def on_item_added(event_or_msg):
        # Depending on version, this might be an Event object OR the Message directly.
        # We handle both cases safely.
        
        # 1. Unwrap the message item
        item = event_or_msg
        if hasattr(event_or_msg, "item"):
            item = event_or_msg.item
        
        # 2. Extract content
        role = getattr(item, "role", "unknown")
        content = getattr(item, "content", "")
        
        # 3. Convert list content to string (common in some LLM outputs)
        if isinstance(content, list):
            content = " ".join([str(x) for x in content])
            
        # 4. Log it
        if content and isinstance(content, str):
            log_line = f"{role.capitalize()}: {content}"
            sdr_agent.transcript.append(log_line)
            logger.info(f"📝 Transcript: {log_line}")

    # --- ON DISCONNECT: SAVE CRM NOTES ---
    @ctx.room.on("disconnected")
  # --- ON DISCONNECT: SAVE CRM NOTES ---
    @ctx.room.on("disconnected")
    def on_room_disconnect(reason):
        logger.info("Call ended. Processing CRM notes...")
        
        # FIX: Save as a List, not a joined String
        transcript_data = sdr_agent.transcript 
        
        if not transcript_data:
            transcript_data = ["(No conversation items were captured)"]

        crm_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "call_summary": {
                "status": "Completed",
                "transcript": transcript_data  # <--- Now it saves the list directly!
            }
        }
        
        with open("crm_notes.json", "a") as f:
            f.write(json.dumps(crm_entry, indent=2) + ",\n")
            
        logger.info("✅ CRM Notes saved to crm_notes.json")

    await session.start(
        agent=sdr_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))