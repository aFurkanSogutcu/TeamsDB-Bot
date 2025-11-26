from typing import Any, Dict, List
from collections import OrderedDict
import time

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