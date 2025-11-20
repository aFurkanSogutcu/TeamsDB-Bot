import os
import json
import re
from typing import Any, Dict, List, Optional

import asyncpg
from dotenv import load_dotenv

from collections import OrderedDict
import time

from datetime import datetime, date
from decimal import Decimal

def _json_default(o):
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, Decimal):
        return float(o)
    return str(o)

# Oturum başına saklayacağımız şeyler
class SessionState:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []   # OpenAI mesaj geçmişi
        self.scratch: Dict[str, Any] = {}          # yapılandırılmış bellek (ör. last_product)
        self.updated_at = time.time()

# Basit LRU (production'da Redis/Cosmos/DB)
class SessionStore:
    def __init__(self, max_sessions=500):
        self.data: OrderedDict[str, SessionState] = OrderedDict()
        self.max_sessions = max_sessions
    def get(self, key: str) -> SessionState:
        st = self.data.get(key)
        if st is None:
            st = SessionState()
            self.data[key] = st
        # LRU touch
        self.data.move_to_end(key)
        st.updated_at = time.time()
        # evict
        while len(self.data) > self.max_sessions:
            self.data.popitem(last=False)
        return st

load_dotenv()

# === Postgres / Common ===
PG_DSN = {
    "user": os.getenv("PGUSER", ""),
    "password": os.getenv("PGPASSWORD", ""),
    "database": os.getenv("PGDATABASE", ""),
    "host": os.getenv("PGHOST", ""),
    "port": int(os.getenv("PGPORT", "")),
}
MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))

def _normalize_sql(sql: str) -> str:
    s = (sql or "").strip()
    # Sondaki gereksiz noktalı virgülleri temizle
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s

def _assert_select_only(sql: str):
    s = _normalize_sql(sql)
    u = s.strip().upper()
    if not u.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")
    for banned in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"):
        if banned in u:
            raise ValueError("Read-only: prohibited statement detected.")

async def run_select(pool: asyncpg.Pool, sql: str, top: Optional[int] = None) -> Dict[str, Any]:
    clean_sql = _normalize_sql(sql)
    _assert_select_only(clean_sql)
    hard_limit = min(int(top or MAX_ROWS), MAX_ROWS)
    wrapped = f"SELECT * FROM ({clean_sql}) AS _mcp_sub LIMIT {hard_limit}"
    async with pool.acquire() as conn:
        print("SQL FROM LLM:", sql)
        print("CLEAN SQL:", clean_sql)
        print("WRAPPED SQL:", wrapped)
        rows = await conn.fetch(wrapped)
        data = [dict(r) for r in rows]
        return {"rowCount": len(data), "rows": data}
    

SCHEMA_NAMES_SQL = """
SELECT n.nspname AS schema_name
FROM pg_catalog.pg_namespace n
WHERE n.nspname NOT IN ('pg_catalog','information_schema')
ORDER BY n.nspname;
"""

SCHEMA_TABLES_SQL = """
SELECT n.nspname AS schema_name, c.relname AS table_name, c.relkind
FROM pg_catalog.pg_namespace n
JOIN pg_catalog.pg_class c ON c.relnamespace = n.oid
WHERE n.nspname NOT IN ('pg_catalog','information_schema')
  AND c.relkind IN ('r','p','v','m')  -- r: table, p: partitioned table, v: view, m: matview
ORDER BY n.nspname, c.relname;
"""

SCHEMA_COLUMNS_SQL = """
SELECT
    table_schema,
    table_name,
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema NOT IN ('pg_catalog','information_schema')
ORDER BY table_schema, table_name, ordinal_position;
"""

async def get_schema_tree_slim(pool: asyncpg.Pool,
                               include_views: bool = False,
                               schema_whitelist: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Şema + tablo + kolon + veri tipi döner.
    Dönen yapı:
      {
      "schemas": {
        "production": {
          "product": [
            {"name": "productid", "type": "integer"},
            {"name": "name", "type": "character varying"},
            ...
    """
    sql_tables = SCHEMA_TABLES_SQL
    if not include_views:
        # sadece 'r' ve 'p' (tablo / partitioned tablo)
        sql_tables = sql_tables.replace("IN ('r','p','v','m')", "IN ('r','p')")

    async with pool.acquire() as conn:
        tables = await conn.fetch(sql_tables)
        columns = await conn.fetch(SCHEMA_COLUMNS_SQL)

    # 2) Boş iskelet: schema -> table -> []
    result: Dict[str, Dict[str, List[Dict[str, str]]]] = {}

    for t in tables:
        schema_name = t["schema_name"]
        table_name = t["table_name"]

        if schema_whitelist and schema_name not in schema_whitelist:
            continue

        result.setdefault(schema_name, {})
        result[schema_name].setdefault(table_name, [])

    # 3) Kolonları uygun tabloya doldur
    for c in columns:
        schema_name = c["table_schema"]
        table_name = c["table_name"]

        # Yukarıda oluşturduğumuz tablo setinde yoksa geç
        if schema_name not in result or table_name not in result[schema_name]:
            continue

        col_name = c["column_name"]
        col_type = c["data_type"]

        result[schema_name][table_name].append(
            {"name": col_name, "type": col_type}
        )

    # 4) Whitelist varsa, en azından boş şema anahtarı olsun
    if schema_whitelist:
        for s in schema_whitelist:
            result.setdefault(s, result.get(s, {}))

    return {"schemas": result}

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

# === Bot Framework =====
from botbuilder.core import (
    TurnContext,
    MessageFactory,
)
from botbuilder.schema import HeroCard, CardAction, ActionTypes, Activity
from botbuilder.core import CardFactory
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication

class DefaultConfig:
    """ Bot Configuration """

    PORT = 3978
    APP_ID = os.environ.get("MICROSOFT_APP_ID", "")
    APP_PASSWORD = os.environ.get("MICROSOFT_APP_PASSWORD", "")
    APP_TYPE = os.environ.get("MICROSOFT_APP_TYPE", "")
    APP_TENANTID = os.environ.get("MICROSOFT_APP_TENANT_ID", "")
CONFIG = DefaultConfig()
ADAPTER = CloudAdapter(ConfigurationBotFrameworkAuthentication(CONFIG))

async def send_buttons_in_bubble(turn_context, text: str, suggestions: List[str]):
    card = HeroCard(
        text=text,
        buttons=[CardAction(type=ActionTypes.im_back, title=s[:40], value=s) for s in suggestions[:3]]
    )
    attachment = CardFactory.hero_card(card)
    reply = MessageFactory.attachment(attachment)
    await turn_context.send_activity(reply)

class McpQueryBot:
    def __init__(self, app):
        self.app = app

    async def on_message_activity(self, turn_context: TurnContext):
        user_text = (turn_context.activity.text or "").strip()

        channel = (turn_context.activity.channel_id or "default").strip()
        conv_id = f"{channel}:{(turn_context.activity.conversation.id or 'default').strip()}"

        sessions: SessionStore = self.app["sessions"]
        session = sessions.get(conv_id) 
        if session is None:
            session = sessions.create(conv_id)  
        if "scratch" not in session.__dict__:
            session.scratch = {}
        if "messages" not in session.__dict__:
            session.messages = []

        raw_name = (getattr(turn_context.activity.from_property, "name", None) or "").strip()
        raw_id   = (getattr(turn_context.activity.from_property, "id", None) or "").strip()
        captured = raw_name or raw_id or "Misafir"

        if not session.scratch.get("user_name") or session.scratch.get("user_name") in {"", "Misafir"}:
            session.scratch["user_name"] = captured

        print("BF From:", {"name": raw_name, "id": raw_id, "kept": session.scratch["user_name"]})

        if user_text.lower() in {"reset", "/reset"}:
            keep_name = session.scratch.get("user_name")
            session.messages.clear()
            session.scratch.clear()
            if keep_name:
                session.scratch["user_name"] = keep_name
            await turn_context.send_activity("Oturum temizlendi.")
            return

        try:
            answer, used_tool = await chat_with_mcp(
                self.app,
                user_text if user_text else "Merhaba",
                session.messages,
                session.scratch,
            )
        except Exception as e:
            await turn_context.send_activity(MessageFactory.text(f"Hata: {e}"))
            return

        if used_tool:
            suggestions = await generate_suggestions(user_text, answer)
            await send_buttons_in_bubble(turn_context, answer, suggestions)
        else:
            await turn_context.send_activity(MessageFactory.text(answer))


from aiohttp import web
from botbuilder.schema import ErrorResponseException

app = web.Application()

import ssl

async def health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})

async def on_startup(app: web.Application):
    ssl_ctx = ssl.create_default_context()

    app["pg_pool"] = await asyncpg.create_pool(
        min_size=1,
        max_size=5,
        ssl=ssl_ctx,
        **PG_DSN,
    )

    app["sessions"] = SessionStore(max_sessions=500)
    app["bot"] = McpQueryBot(app)

    print("[startup] PG pool created")

    async with app["pg_pool"].acquire() as conn:
        n = await conn.fetchval("SELECT COUNT(*) FROM production.product;")
        print("product count:", n)

async def on_shutdown(app: web.Application):
    await app["pg_pool"].close()
    print("[shutdown] PG pool closed")

async def mcp_db_schema(request: web.Request) -> web.Response:
    app = request.app
    q = request.rel_url.query

    include_views = q.get("include_views", "false").lower() in {"1", "true", "yes"}
    schemas = q.get("schemas")
    wl = [s.strip() for s in schemas.split(",")] if schemas else None

    res = await get_schema_tree_slim(
        app.state.pg_pool,
        include_views=include_views,
        schema_whitelist=wl,
    )
    return web.json_response(res)

class MyBot:
    async def on_turn(self, context: TurnContext):
        try:
            if context.activity.type == "message":
                await context.send_activity("pong")
            else:
                await context.send_activity(f"Activity type: {context.activity.type}")
        except ErrorResponseException as e:
            print("Reply failed:", e)          # status code
            if hasattr(e, "response") and e.response is not None:
                print("Raw response:", e.response.text)
            raise

async def messages(request: web.Request) -> web.Response:
    auth_header = request.headers.get("Authorization", "")
    body = await request.json()

    activity = Activity().deserialize(body)

    bot: McpQueryBot = request.app["bot"]

    async def aux_func(turn_context: TurnContext):
        if turn_context.activity.type == "message":
            await bot.on_message_activity(turn_context)
        else:
            await turn_context.send_activity(f"Activity type: {turn_context.activity.type}")

    await ADAPTER.process_activity(auth_header, activity, aux_func)

    return web.Response(status=200)

# lifecycle hook'ları EKLEe
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

app.router.add_get("/health", health)
app.router.add_post("/api/messages", messages)
 
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=3978)