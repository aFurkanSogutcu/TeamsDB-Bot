import os
from typing import List
from dotenv import load_dotenv
from memory import SessionStore, SessionState
from model import chat_with_mcp, generate_suggestions
# === Bot Framework =====
from botbuilder.core import (
    TurnContext,
    MessageFactory,
)
from botbuilder.schema import HeroCard, CardAction, ActionTypes
from botbuilder.core import CardFactory
from botbuilder.integration.aiohttp import CloudAdapter, ConfigurationBotFrameworkAuthentication

load_dotenv()

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
