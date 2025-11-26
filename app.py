import os
import asyncpg
from dotenv import load_dotenv
from aiohttp import web
from botbuilder.schema import Activity
import ssl
from mcp_func import get_schema_tree_slim
from memory import SessionStore
from bot import McpQueryBot
from botbuilder.core import TurnContext
from bot import ADAPTER

load_dotenv()

# === Postgres / Common ===
PG_DSN = {
    "user": os.getenv("PGUSER", ""),
    "password": os.getenv("PGPASSWORD", ""),
    "database": os.getenv("PGDATABASE", ""),
    "host": os.getenv("PGHOST", ""),
    "port": int(os.getenv("PGPORT", "")),
}

app = web.Application()

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

# lifecycle hook'ları EKLE
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

app.router.add_get("/health", health)
app.router.add_post("/api/messages", messages)
 
if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=3978)