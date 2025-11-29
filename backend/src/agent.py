import logging
from typing import Annotated, Optional
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
from merchant import MerchantAPI

load_dotenv(".env.local")
logger = logging.getLogger("agent")

# Initialize the Merchant System
merchant = MerchantAPI()

class CommerceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'ShopBot', an AI Shopping Assistant powered by the Agentic Commerce Protocol.
            
            YOUR JOB:
            Help users browse the catalog, find deals, and place orders.
            
            CAPABILITIES:
            1. Browse: Use `search_catalog` to find items.
               - If a user asks for "hoodies", search for "hoodie".
               - If they ask for "electronics", use category="electronics".
               - Always call the search_catalog function when users ask about products.
            
            2. Buy: Use `place_order` when the user confirms a purchase.
               - Ask for the quantity if not specified.
               - Ask for their name if you don't know it.
            
            3. History: Use `check_last_order` if they ask "What did I buy?".
            
            IMPORTANT: Always use the tools/functions available to you. Don't just talk about products - actually search for them.
            
            TONE: Efficient, polite, and transactional.
            """,
        )

    # --- TOOL 1: BROWSE ---
    @function_tool
    async def search_catalog(
        self, 
        query: Annotated[str, "Search keyword (e.g. 'hoodie', 'keyboard', 'mug')"] = "",
        category: Annotated[Optional[str], "Category filter: 'clothing', 'electronics', or 'home'"] = None,
        max_price: Annotated[Optional[str], "Maximum price filter in dollars"] = None
    ):
        """Search for products in the catalog. Use this whenever a user asks about products."""
        # Handle optional parameters
        if category is None:
            category = ""
        if max_price is None or max_price == "":
            max_price = "10000"
        
        logger.info(f"🔎 Searching: q='{query}' cat='{category}' price='{max_price}'")
        
        results = merchant.search_products(query, category, max_price)
        
        if not results:
            return "No products found matching those criteria."
        
        summary = "Found these items:\n"
        for p in results:
            attrs = p.get('attributes', {})
            attr_str = ", ".join([f"{k}: {v}" for k,v in attrs.items()])
            summary += f"- {p['name']} (${p['price']}) [{attr_str}]\n"
            
        return summary

    # --- TOOL 2: BUY ---
    @function_tool
    async def place_order(
        self, 
        product_name: Annotated[str, "Name of product to buy"],
        customer_name: Annotated[str, "Name of the customer"],
        quantity: Annotated[int, "Quantity to purchase"] = 1
    ):
        """Places an order and saves it to the backend."""
        logger.info(f"🛒 Ordering: {quantity}x {product_name} for {customer_name}")
        
        result = merchant.create_order(product_name, quantity, customer_name)
        
        if "error" in result:
            return f"Error: {result['error']}"
        
        return f"✅ Order Confirmed! ID: {result['order_id']}. Total: ${result['total_amount']} {result['currency']}."

    # --- TOOL 3: HISTORY ---
    @function_tool
    async def check_last_order(self):
        """Retrieves the details of the last order placed."""
        order = merchant.get_last_order()
        if not order:
            return "You haven't placed any orders yet."
            
        item = order['items'][0]
        return f"Your last order was for {item['quantity']}x {item['name']} on {order['timestamp']}. Total: ${order['total_amount']}."

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),  # Using latest Gemini model
        tts=murf.TTS(
            voice="en-US-terrell",
            style="Promo",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        agent=CommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))