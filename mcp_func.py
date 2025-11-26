import os
from typing import Any, Dict, List, Optional
import asyncpg

MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))

# POST SQL MCP #
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
    

# GET DB SCHEMA #
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