# sql_data_agent

A multi-agent, LangGraph-based system that answers natural-language questions over a PostgreSQL database and runs ETL pipelines — built on top of a ride-sharing dataset (users, drivers, vehicles, rides, payments, ratings).

A router agent classifies each incoming request as either a **SQL** question or an **ETL** task and dispatches it to the corresponding specialist agent.

## Architecture

```
                 ┌──────────────┐
   user request →│  Router Node │
                 └──────┬───────┘
                        │ classifies via RouterSchema (LLM structured output)
             ┌──────────┴──────────┐
             ▼                     ▼
      ┌─────────────┐       ┌─────────────┐
      │  SQL Node    │       │  ETL Node    │
      │ sql_analyst  │       │ etl_analyst  │
      └─────────────┘       └─────────────┘
             │                     │
        query Postgres      extract / transform
        & return answer     data via pandas
```

**`agents/data_agent.py`** — top-level graph. A router node classifies the request (`RouterSchema`: `sql` or `etl`) and conditionally routes to the SQL or ETL node.

**`agents/sql_analyst.py`** — converts a natural-language question into a safe, executed SQL query, in 7 steps:
1. Curate/refine the user's question via LLM
2. Build context — pull table schema + sample rows from Postgres
3. Generate SQL from the curated question
4. Safety check — an LLM judge (`JudgeSchema`) rejects queries containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `TRUNCATE`, `CREATE`
5. Conditional routing — execute if safe, reject otherwise
6. Execute the query against PostgreSQL (results default-limited to 10 rows unless the question says otherwise)
7. Represent the result as a user-friendly answer

**`agents/etl_analyst.py`** — a tool-using agent loop (message → LLM → tool call → tool result → LLM …) built on two tools:
- extract data from an API endpoint and save as CSV / JSON / Parquet
- transform an existing file by generating and executing pandas code on the fly
Tool results are summarized before being added back to message history, to keep context small.

**`models/schema.py`** — Pydantic state schemas shared across the graphs: `AgentSchema`, `JudgeSchema`, `EtlAgentSchema`, `RouterSchema`, `DataAgentSchema`.

Graph diagrams are checked into the repo as PNGs: `data_agent_graph.png`, `sql_analyst_graph.png`, `etl_analyst_graph.png`.

## Tech Stack

| Component | Library |
|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) >= 1.2.11, [LangChain](https://github.com/langchain-ai/langchain) >= 1.4.0 |
| LLM provider | Groq, via `langchain_groq.ChatGroq` (model: `openai/gpt-oss-120b`) |
| Database | PostgreSQL, via `psycopg2-binary` >= 2.9.12 |
| Data handling | pandas >= 3.0.5 |
| Config | `python-dotenv` |
| Runtime / packaging | Python >= 3.11, [uv](https://github.com/astral-sh/uv) (`uv_build`) |

## Project Structure

```
sql_data_agent/
├── agents/
│   ├── data_agent.py       # Router graph → dispatches to sql_analyst / etl_analyst
│   ├── sql_analyst.py      # NL question → safe SQL → execution → answer
│   └── etl_analyst.py      # Tool-calling ETL agent (extract / transform)
├── models/
│   └── schema.py           # Pydantic state schemas for all graphs
├── utils/
│   ├── database.py         # DatabaseUtil: psycopg2 connection, schema introspection, execute_sql
│   ├── etl_tools.py        # ETLTools: extract_load, transform_load_context, execute_code
│   └── llm_pick.py         # pick_llm(level): returns a configured ChatGroq instance
├── src/data_agent/
│   └── __init__.py         # CLI entry point (`data-agent` command)
├── data/                   # Source CSVs for the ride-sharing dataset + extract/transform scratch space
│   ├── users.csv / vehicles.csv / rides.csv / payments.csv / ratings.csv
│   └── extract/…           # ETL agent output/scratch (e.g. API extracts, transformed files)
├── feed_db.py               # One-shot script: creates schema + loads CSVs into PostgreSQL
├── test_schema_details.txt  # Sample output of DatabaseUtil.schema_details()
├── main.py
├── data_agent_graph.png / sql_analyst_graph.png / etl_analyst_graph.png
├── pyproject.toml
└── uv.lock
```

## Database Schema

`feed_db.py` provisions five tables in the `public` schema of PostgreSQL:

**users** — `user_id` PK, `first_name`, `last_name`, `email` (unique), `phone`, `city`, `province`, `user_type`, `signup_date`, `is_active`

**vehicles** — `vehicle_id` PK, `driver_id` → `users.user_id`, `make`, `model`, `year`, `license_plate` (unique), `color`, `is_active`

**rides** — `ride_id` PK, `rider_id` / `driver_id` → `users.user_id`, `requested_at`, `pickup_time`, `dropoff_time`, pickup/dropoff lat-long, `distance_km`, `fare`, `surge_multiplier`, `status`, `cancellation_reason`

**payments** — `payment_id` PK, `ride_id` → `rides`, `user_id` → `users`, `amount`, `payment_method`, `payment_status`, `transaction_id` (unique), `payment_time`

**ratings** — `rating_id` PK, `ride_id` → `rides`, `rider_id` / `driver_id` → `users`, `rating` (1–5, checked), `comment`, `rated_at`

## Setup

**1. Install dependencies**
```bash
uv sync
```

**2. Configure environment variables** (a `.env` file is loaded via `python-dotenv`)
```bash
# PostgreSQL (used by feed_db.py, utils/database.py)
host=<db-host>
port=5432          # optional, defaults to 5432
database=<db-name>
user=<db-user>
password=<db-password>

# LLM
GROQ_API_KEY=<your-groq-api-key>
```

**3. Load the ride-sharing dataset into PostgreSQL**
```bash
python feed_db.py
```
This creates the five tables above and bulk-loads the matching CSVs from `data/` via `COPY`.

**4. Run the agent**
```bash
uv run data-agent
```

## Example Usage

The SQL analyst agent (as demonstrated in `agents/sql_analyst.py`) can take a question like:

> "What are the most common payment methods?"

and produce the curated question → generated SQL → safety verdict → executed result → natural-language answer.

The ETL analyst agent (as demonstrated in `agents/etl_analyst.py`) can handle a request like:

> "Extract Pokémon data from the PokéAPI and save it as CSV, then filter it to just Bulbasaur."

by calling its extract and transform tools in sequence.

