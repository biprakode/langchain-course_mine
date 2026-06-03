from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage , ToolMessage
from langsmith import traceable
from langchain_groq import ChatGroq

MODEL="llama-3.3-70b-versatile"
MAX_ITERATIONS = 10

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product.
    :param product: The name of the product to search for.
    :return: The price as a float, or -1.0 if not found.
    """
    print(f"Looking up the price of: {product}")

    # Giant mock database for testing LLM tool call variability
    prices = {
        # --- Electronics & Tech ---
        "iphone 15 pro max": 1199.99,
        "iphone 15": 799.00,
        "samsung galaxy s24 ultra": 1299.99,
        "google pixel 8 pro": 999.00,
        "macbook air m3": 1099.00,
        "macbook pro m3 max": 3199.00,
        "sony wh-1000xm5": 398.00,
        "airpods pro 2": 249.00,
        "nintendo switch oled": 349.99,
        "playstation 5 slim": 499.99,
        "xbox series x": 499.00,
        "asus rog ally": 699.99,
        "ipad air": 599.00,
        "kindle paperwhite": 149.99,

        # --- Apparel & Footwear ---
        "nike air max 90": 130.00,
        "adidas ultraboost light": 190.00,
        "levis 501 original jeans": 79.50,
        "patagonia torrentshell jacket": 179.00,
        "carhartt wip beanie": 28.00,
        "birkenstock arizona sandals": 130.00,

        # --- Smart Home & Appliances ---
        "dyson v15 detect vacuum": 749.99,
        "instant pot duo plus": 129.95,
        "philips hue starter kit": 199.99,
        "irobot roomba j7+": 799.99,
        "keurig k-elite coffee maker": 189.99,
        "sonos era 100": 249.00,
        "amazon echo dot": 49.99,

        # --- Everyday Groceries & Essentials ---
        "avocado": 1.25,
        "organic milk 1 gallon": 5.99,
        "whole wheat bread": 3.49,
        "eggs 1 dozen": 4.29,
        "starbucks dark roast coffee beans": 14.95,
        "olive oil 500ml": 12.99,
        "sriracha hot sauce": 6.50,
        "whey protein powder 2lbs": 34.99,

        # --- Fitness & Outdoor Gear ---
        "hydro flask 32 oz water bottle": 44.95,
        "yoga mat 6mm": 25.00,
        "bowflex selecttech adjustable dumbbells": 429.00,
        "fitbit charge 6": 159.95,
        "apple watch ultra 2": 799.00,
        "osprey talon 22 backpack": 160.00
    }

    search_query = product.lower().strip()
    return prices.get(search_query, -1.0)

@tool
def apply_discount(price : float , discount_tier : str)-> float:
    """Applies the discount to the price, based on the discount tier.
    Available discount tiers are: BRONZE, SILVER, GOLD"""

    print("Applying discount for " + str(price) , discount_tier)
    discounts = {"bronze" : 0.05 , "silver" : 0.10 , "gold" : 0.15}
    discount = discounts.get(discount_tier.strip().lower())
    return round(price * (1 - discount), 2)

@traceable(name = "Langchain agent loop")
def run_agent(question : str):
    tools = [get_product_price , apply_discount]
    tools_dict = {t.name : t for t in tools}

    #llm = init_chat_model(f"ollama:{MODEL}" , temperature = 0)
    llm = ChatGroq(model=MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print("Question: " + question)
    print("=" * 60)

    messages = [
        SystemMessage(
            content=("You are a strict, precise retail price calculation assistant. "
            "Your objective is to calculate product prices with applied discounts.\n\n"
    
            "CRITICAL INSTRUCTIONS:\n"
            "1. NEVER guess, assume, or hallucinate the price of a product or a discount amount. "
            "You do not know any prices or discount percentages out of the box.\n"
            "2. You MUST use the `get_product_price` tool to fetch the price of any requested item. "
            "Do not try to answer from internal knowledge.\n"
            "3. You MUST use the `apply_discount` tool if a discount tier (BRONZE, SILVER, GOLD) is mentioned. "
            "Do not calculate math or percentages manually.\n"
            "4. STRICT ERROR HANDLING: If `get_product_price` returns -1.0, it means the product is missing "
            "from your inventory database. You MUST immediately stop execution and state verbatim: "
            "'Error: The requested product is out of stock or does not exist in the catalog.' "
            "Do not make up a value if you see -1.0.\n"
            "5. Stick purely to the operational facts returned by your tools. Keep your ultimate answer concise.")
        ),
        HumanMessage(content=question)
    ]

    for iter in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final answer = " + str(ai_message.content))
            break

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        tool_id = tool_call.get("id")

        print("Tool selected: " + tool_name + " args = " + str(tool_args))

        tool_use = tools_dict.get(tool_name)
        if tool_use is None:
            raise("Tool not found = " + tool_name)

        obs = tool_use.invoke(tool_args)
        print("Tool result = " + str(obs))

        messages.append(ai_message)
        messages.append(ToolMessage(content=str(obs), tool_call_id=tool_id))

    print("MAX ITERATIONS = ", MAX_ITERATIONS)

run_agent("How much does a whole wheat bread cost?")

print("=" * 60)

run_agent("Can you check the price of a sony wh-1000xm5 with a SILVER discount?")

print("=" * 60)

run_agent("I want to buy a dyson v15 detect vacuum and I have a GOLD discount tier. What is my final price?")

print("=" * 60)

run_agent("What is the price of a Tesla Model 3 with a GOLD discount?")

print("=" * 60)

run_agent("A product costs $100. Apply a GOLD discount to it.")