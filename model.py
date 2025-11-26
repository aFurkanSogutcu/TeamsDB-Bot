import os
import json
import re
from typing import Any, Dict, List
from datetime import datetime, date
from decimal import Decimal
from dotenv import load_dotenv
from mcp_func import run_select, get_schema_tree_slim

def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)

load_dotenv()

# === OpenAI / Tools ===
from openai import OpenAI
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE") or None
assert OPENAI_API_KEY, "Please set OPENAI_API_KEY"
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

# SCHEMA (for DB questions only):
# person.address: [addressid, addressline1, city, stateprovinceid, postalcode, spatiallocation, rowguid, modifieddate]
# person.phonenumbertype: [phonenumbertypeid, name, modifieddate]
# production.product: [productid, name, productnumber, makeflag, finishedgoodsflag, color, safetystocklevel, reorderpoint, standardcost, listprice, size, sizeunitmeasurecode, weightunitmeasurecode, weight, daystomanufacture, productline, class, style, productsubcategoryid, productmodelid, sellstartdate, sellenddate, discontinueddate, rowguid, modifieddate]
# production.productcategory: [productcategoryid, name, rowguid, modifieddate]
# sales.salesorderdetail: [salesorderid, salesorderdetailid, carriertrackingnumber, orderqty, productid, specialofferid, unitprice, unitpricediscount, rowguid, modifieddate]

SYSTEM_PROMPT = """\
Sen Türkçe konuşan, doğal sohbet edebilen; yalnızca gerektiğinde AdventureWorks veritabanına
sql_query aracıyla SELECT sorguları yapan akıllı bir asistansın.

BEHAVIOR:
- AdventureWorks ile ilgili veri sorularında sql_query aracını kullan. Tablo/sütun adlarından emin değilsen önce db_schema aracını çağırıp şemayı öğren.
- Talk with user in Turkish language
- If the user greets you or asks general knowledge questions (not about AdventureWorks data), answer directly. Do NOT call tools.
- Use the sql_query tool only for AdventureWorks database questions (facts, counts, product details, prices, colors, categories, sales, orders).
- Never ask generic filler like "Do you have any other questions?" unless the user seems done; always attempt a helpful answer first.
- When generating SQL, use ONLY the exact schema-qualified names below, and use ILIKE for text filters.
- Read-only only (SELECT/CTE). Never modify data.
- Always answer in English, and do not show SQL unless the user explicitly asks.

FEW-SHOTS:
- USER: "Hello, how are you?"
  ASSISTANT: "Hi! I’m doing well — how can I help today?"

- USER: "Which city has the largest area in Turkey?"
  ASSISTANT: "Konya is the largest by land area (province level), about 40,800 km²."

- USER: "How many black products are there?"
  ASSISTANT (SQL): SELECT COUNT(*) AS count FROM production.product WHERE color ILIKE 'black';
""" 
    
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "db_schema",
            "description": "Return database skeleton: schemas, tables, and for each table its columns and data types (no row data).",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_views": {
                        "type": "boolean",
                        "description": "Include views/materialized views",
                        "default": False
                    },
                    "schemas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional whitelist of schema names"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sql_query",
            "description": "Run a read-only SELECT on AdventureWorks. Returns JSON rows.",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}, "top": {"type": "integer"}},
                "required": ["sql"]
            }
        }
    }
]


MAX_TURNS = 20
def _clip_history(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return msgs[-MAX_TURNS:]

async def chat_with_mcp(app, user_prompt: str,
                        history: List[Dict[str, Any]],
                        scratch: Dict[str, Any]) -> tuple[str, bool]:
    user_name = scratch.get("user_name") or "Misafir"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Current user name: {user_name}"},
        *_clip_history(history),
        {"role": "user", "content": user_prompt},
    ]

    first = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        tools=TOOLS,
        tool_choice="auto",
        messages=messages,
    )
    msg = first.choices[0].message

    assistant_msg: Dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        print("msg.tool_calls:", msg.tool_calls)
        assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
    messages.append(assistant_msg)

    used_sql_query = False
    tool_msgs_all: List[Dict[str, Any]] = []

    if msg.tool_calls:
        for tc in (msg.tool_calls or []):
            fname = getattr(tc.function, "name", "")
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}

            if fname == "db_schema":
                include_views = bool(args.get("include_views", False))
                schemas = args.get("schemas") or None
                schema_json = await get_schema_tree_slim(
                    app["pg_pool"],
                    include_views=include_views,
                    schema_whitelist=schemas
                )
                #print("DB SCHEMA JSON:", json.dumps(schema_json, indent=2, ensure_ascii=False))

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "db_schema",
                    "content": json.dumps(schema_json, ensure_ascii=False, default=_json_default),
                }
                messages.append(tool_msg)
                tool_msgs_all.append(tool_msg)   

            elif fname == "sql_query":
                sql = (args.get("sql") or "")
                top = args.get("top")
                res = await run_select(app["pg_pool"], sql, top)

                scratch["last_rows"] = res.get("rows", [])
                if len(scratch["last_rows"]) == 1:
                    scratch["last_product"] = scratch["last_rows"][0]

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": "sql_query",
                    "content": json.dumps(res, ensure_ascii=False, default=_json_default),
                }
                messages.append(tool_msg)
                tool_msgs_all.append(tool_msg)   
                used_sql_query = True

            else:
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fname or "unknown",
                    "content": json.dumps({"error": f"Unknown tool: {fname}"}, ensure_ascii=False),
                }
                messages.append(tool_msg)
                tool_msgs_all.append(tool_msg)
            
            print("kullanılan func: ", fname)

    final = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=messages,
    )
    answer = final.choices[0].message.content or "Boş yanıt."

    history.append({"role": "user", "content": user_prompt})
    history.append(assistant_msg)

    if assistant_msg.get("tool_calls"):
        history.extend(tool_msgs_all)
        if len(tool_msgs_all) < len(assistant_msg["tool_calls"]):
            assistant_msg.pop("tool_calls", None)

    history.append({"role": "assistant", "content": answer})
    if len(history) > MAX_TURNS:
        del history[:-MAX_TURNS]

    return answer, used_sql_query

# === Suggestions LLM ==
DB_SCHEMA_TEXT = """\
Tables and columns (exact):
person.address: [addressid, addressline1, city, stateprovinceid, postalcode, spatiallocation, rowguid, modifieddate]
person.phonenumbertype: [phonenumbertypeid, name, modifieddate]
production.product: [productid, name, productnumber, makeflag, finishedgoodsflag, color, safetystocklevel, reorderpoint, standardcost, listprice, size, sizeunitmeasurecode, weightunitmeasurecode, weight, daystomanufacture, productline, class, style, productsubcategoryid, productmodelid, sellstartdate, sellenddate, discontinueddate, rowguid, modifieddate]
production.productcategory: [productcategoryid, name, rowguid, modifieddate]
sales.salesorderdetail: [salesorderid, salesorderdetailid, carriertrackingnumber, orderqty, productid, specialofferid, unitprice, unitpricediscount, rowguid, modifieddate]
"""

SUGGESTIONS_SYS = f"""\
You generate short follow-up button suggestions for a Turkish SQL helper chat over AdventureWorks.

HARD RULES:
- Stick to ONLY these tables/columns and concepts (do not invent new ones):
{DB_SCHEMA_TEXT}
- Output ONLY JSON: {{"suggestions":[ "...", "..." ]}}
- 3 items, Turkish, ≤ 40 chars each.
- Each item must be a realistic follow-up the user could click (do not output SQL, only natural-language prompts).
- Prefer concrete intents that map to the schema (renk/color, kategori/category, fiyat/price, satış/sales).
- If context is vague, produce generic but schema-relevant queries (örn: "Renk bazlı ürün sayıları", "En çok satan 10 ürün").
"""

async def generate_suggestions(user_prompt: str, answer_text: str) -> List[str]:
    try:
        comp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": SUGGESTIONS_SYS},
                {"role": "user", "content": json.dumps({
                    "user_prompt": user_prompt,
                    "answer_text": answer_text
                }, ensure_ascii=False)}
            ]
        )
        raw = comp.choices[0].message.content or "{}"
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group(0) if m else raw)
        items = data.get("suggestions", [])
        cleaned = []
        for it in items:
            if not isinstance(it, str):
                continue
            t = it.strip().strip('"').strip()
            if 1 <= len(t) <= 60 and t not in cleaned:
                cleaned.append(t)
        if cleaned:
            return cleaned[:3]
    except Exception:
        pass
    return [
        "En çok satan 10 ürün",
        "Renk bazlı ürün sayıları",
        "Ürün kategorilerini listele"
    ]