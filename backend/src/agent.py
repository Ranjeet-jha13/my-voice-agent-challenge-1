import logging
import json
import os
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

# --- GAME STATE CLASS ---
class GameState:
    def __init__(self):
        self.hp = 100
        self.inventory = ["Signal Flare", "Rusty Blade"]
        self.location = "Outer Wall"
        self.quest = "Survive"

    def to_json(self):
        return json.dumps({
            "hp": self.hp,
            "inventory": self.inventory,
            "location": self.location,
            "quest": self.quest
        }, indent=2)

class GameMaster(Agent):
    def __init__(self):
        self.game = GameState()
        
        # Save initial state immediately when game starts
        self._save_game_state()
        
        super().__init__(
            instructions=f"""You are a Scripted Game Master for an Attack on Titan demo.
            
            Your job is to follow this EXACT script sequence. Do not deviate.
            
            SCENE 1 (Start):
            - Describe the windy Outer Wall.
            - Ask: "What do you do?"
            
            SCENE 2 (Trigger: User says "Scan" or "Look"):
            - Say: "You spot a 15-meter Titan gripping the wall! It sees you."
            - Ask: "What do you do?"
            
            SCENE 3 (Trigger: User says "Flare" or "Signal"):
            - CALL TOOL: `update_game_state` with ALL FOUR parameters: damage_taken=0, new_location="", item_found="", item_used="Signal Flare".
            - Say: "The red flare shoots up! But the Titan swipes at you!"
            - Ask: "What do you do?"
            
            SCENE 4 (Trigger: User says "Attack" or "Blade"):
            - CALL TOOL: `update_game_state` with ALL FOUR parameters: damage_taken=20, new_location="", item_found="", item_used="Rusty Blade".
            - Say: "You slash its hand, but the steam burns you! You take 20 damage."
            - Ask: "What do you do?"
            
            SCENE 5 (Trigger: User says "Retreat" or "Inner Keep"):
            - CALL TOOL: `update_game_state` with ALL FOUR parameters: damage_taken=0, new_location="Inner Keep", item_found="", item_used="".
            - Say: "You grapple away to the Inner Keep. You are safe. Mission Complete."
            - STOP TALKING.
            
            IMPORTANT: When calling update_game_state, ALWAYS provide all four parameters. Use empty strings ("") or 0 for unused parameters.
            
            Current State: {self.game.to_json()}
            """,
        )

    def _save_game_state(self):
        """Helper method to save game state to JSON file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src
        parent_dir = os.path.dirname(current_dir)  # backend
        
        # Ensure directory exists
        os.makedirs(parent_dir, exist_ok=True)
        
        save_path = os.path.join(parent_dir, "game_state.json")
        current_state_json = self.game.to_json()
        
        try:
            with open(save_path, "w") as f:
                f.write(current_state_json)
            logger.info(f"✅ SAVED JSON TO: {save_path}")
            logger.info(f"📄 Content: {current_state_json}")
        except Exception as e:
            logger.error(f"❌ Failed to save file: {e}")
            logger.error(f"Attempted path: {save_path}")

    @function_tool
    async def update_game_state(
        self,
        damage_taken: Annotated[int, "Amount of HP damage taken"],
        new_location: Annotated[str, "New location name"],
        item_found: Annotated[str, "Item to add to inventory"],
        item_used: Annotated[str, "Item to remove from inventory"]
    ):
        """
        Updates the game state. All four parameters are required (use empty strings or 0 for unused values).
        """
        logger.info(f"🎮 update_game_state CALLED with: damage={damage_taken}, location={new_location}, found={item_found}, used={item_used}")
        
        # 1. Update State Logic
        if damage_taken > 0:
            self.game.hp -= damage_taken
            logger.info(f"💔 Damage taken: {damage_taken}. HP now: {self.game.hp}")

        if new_location and new_location != "":
            self.game.location = new_location
            logger.info(f"📍 Location changed to: {new_location}")

        if item_found and item_found != "":
            self.game.inventory.append(item_found)
            logger.info(f"📦 Item found: {item_found}")

        if item_used and item_used != "" and item_used in self.game.inventory:
            self.game.inventory.remove(item_used)
            logger.info(f"🔥 Item used: {item_used}")

        # 2. Update quest status if at Inner Keep
        if new_location == "Inner Keep":
            self.game.quest = "Mission Complete"
            logger.info(f"🏆 Quest completed!")

        # 3. SAVE TO FILE IMMEDIATELY
        self._save_game_state()

        current_state_json = self.game.to_json()
        return f"STATE UPDATED. Current Status: {current_state_json}"

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    gm_agent = GameMaster()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-terrell", 
            style="Promo",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=gm_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))