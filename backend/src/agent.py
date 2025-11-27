import logging
import json
import os
from datetime import datetime, timedelta
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

# --- 1. LOAD CATALOG ---
try:
    with open("catalog.json", "r") as f:
        CATALOG = json.load(f)
except FileNotFoundError:
    CATALOG = []
    logger.error("CATALOG.JSON NOT FOUND!")

class GrocerAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'FreshBot', an intelligent grocery ordering assistant.
            
            YOUR GOAL:
            1. Help users order food (add/remove items).
            2. Track existing orders using the `track_order` tool.
            
            CATALOG RULES:
            - If a user asks for "ingredients for a sandwich", add Bread, Peanut Butter, and Jelly automatically.
            - If a user asks for "ingredients for pasta", add Pasta and Sauce.
            
            TOOLS:
            - `add_to_cart`: For adding items.
            - `get_cart_status`: To see what is currently selected.
            - `place_order`: ONLY when they say "checkout" or "I'm done".
            - `track_order`: If they ask "Where is my order?" or "Status update".
            
            TONE: Helpful, efficient, and polite.
            """,
        )
        self.cart = [] 

    def _find_item(self, query):
        query = query.lower()
        return [i for i in CATALOG if query in i['name'].lower() or query in i['tags']]

    # --- TOOL 1: ADD TO CART (With Recipe Logic) ---
    @function_tool
    async def add_to_cart(
        self, 
        item_name: Annotated[str, "Item name or recipe"], 
        quantity: Annotated[int, "Quantity"] = 1
    ):
        """Adds items to cart. Handles 'ingredients for X' requests."""
        # Recipe Logic
        if "sandwich" in item_name.lower():
            self.cart.append({"name": "Whole Wheat Bread", "price": 4.50, "qty": 1})
            self.cart.append({"name": "Peanut Butter", "price": 4.00, "qty": 1})
            self.cart.append({"name": "Grape Jelly", "price": 3.50, "qty": 1})
            return "Smart choice! Added Bread, PB, and Jelly for your sandwich."
        
        if "pasta" in item_name.lower():
            self.cart.append({"name": "Spaghetti Pasta", "price": 2.00, "qty": 1})
            self.cart.append({"name": "Marinara Sauce", "price": 4.00, "qty": 1})
            return "Yum! Added Spaghetti and Sauce for pasta night."

        # Standard Logic
        matches = self._find_item(item_name)
        if not matches: return f"Sorry, I don't have '{item_name}'."
        if len(matches) > 1: return f"Did you mean: {', '.join([m['name'] for m in matches])}?"

        target = matches[0]
        self.cart.append({"name": target['name'], "price": target['price'], "qty": quantity})
        return f"Added {quantity} {target['name']} to cart."

    # --- TOOL 2: VIEW CART ---
    @function_tool
    async def get_cart_status(self):
        """Shows current cart items."""
        if not self.cart: return "Your cart is empty."
        items_list = "\n".join([f"- {i['qty']}x {i['name']}" for i in self.cart])
        total = sum(i['price'] * i['qty'] for i in self.cart)
        return f"Cart:\n{items_list}\nTotal: ${total:.2f}"

    # --- TOOL 3: PLACE ORDER (SAVES TO JSON) ---
    @function_tool
    async def place_order(self, customer_name: Annotated[str, "Name"]):
        """Places the order and saves to JSON."""
        if not self.cart: return "Cart is empty!"
        
        total = sum(i['price'] * i['qty'] for i in self.cart)
        # Create Order Data with Timestamp
        order_data = {
            "order_id": f"ORD-{int(datetime.now().timestamp())}",
            "timestamp": datetime.now().isoformat(), # Save time for tracking logic
            "customer": customer_name,
            "items": self.cart,
            "total": total,
            "status": "Received" 
        }

        # Append to orders.json
        file_path = "orders.json"
        history = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f: history = json.load(f)
            except: pass
        
        history.append(order_data)
        with open(file_path, "w") as f: json.dump(history, f, indent=2)
        
        self.cart = [] # Clear cart
        return f"Order placed! ID: {order_data['order_id']}. You can ask me to track it."

    # --- TOOL 4: ADVANCED TRACKING (MOCK TIME LOGIC) ---
    @function_tool
    async def track_order(self):
        """Checks the status of the most recent order based on time passed."""
        file_path = "orders.json"
        if not os.path.exists(file_path): return "No orders found."
        
        try:
            with open(file_path, "r") as f: history = json.load(f)
            if not history: return "No orders found."
            
            # Get latest order
            latest_order = history[-1]
            order_time = datetime.fromisoformat(latest_order["timestamp"])
            time_diff = datetime.now() - order_time
            
            # Mock Logic: Update status based on how much time passed
            new_status = latest_order["status"]
            if time_diff > timedelta(seconds=120): # 2 mins later
                new_status = "Delivered 🏠"
            elif time_diff > timedelta(seconds=60): # 1 min later
                new_status = "Out for Delivery 🛵"
            elif time_diff > timedelta(seconds=30): # 30 secs later
                new_status = "Being Prepared 🍳"
            
            # Save update if changed
            if new_status != latest_order["status"]:
                latest_order["status"] = new_status
                with open(file_path, "w") as f: json.dump(history, f, indent=2)
            
            return f"Order Status for {latest_order['customer']}: {new_status}"
            
        except Exception as e:
            return f"Error tracking order: {e}"

# --- SETUP ---
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Conversation", tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2), text_pacing=True),
        turn_detection=MultilingualModel(),
    )
    await session.start(agent=GrocerAgent(), room=ctx.room, room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()))
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))