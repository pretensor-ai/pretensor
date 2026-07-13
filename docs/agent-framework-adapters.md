# Agent-framework adapters

Pretensor exposes six graph tools — `schema`, `context`, `traverse`, `impact`, `query`, `validate_sql` — as native tool objects for LangChain, LlamaIndex, and Google ADK.

No MCP server process is required. The adapters call the same underlying payload functions the MCP server uses, so output is identical.

## Prerequisites

Index your database first:

```bash
pretensor index postgresql://user:pass@host/mydb
```

---

## LangChain

```bash
pip install 'pretensor[langchain]'
```

```python
from pathlib import Path
from pretensor.integrations.langchain import load_langchain_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

tools = load_langchain_tools(Path(".pretensor"))
llm = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([("system", "You are a data assistant."), ("placeholder", "{agent_scratchpad}")])
agent = AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools)
```

---

## LlamaIndex

```bash
pip install 'pretensor[llama-index]'
```

```python
from pathlib import Path
from pretensor.integrations.llamaindex import load_llamaindex_tools
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

tools = load_llamaindex_tools(Path(".pretensor"))
agent = ReActAgent.from_tools(tools, llm=OpenAI(model="gpt-4o"), verbose=True)
```

---

## Google ADK

```bash
pip install 'pretensor[google-adk]'
```

```python
from pathlib import Path
from pretensor.integrations.google_adk import load_adk_tools
from google.adk.agents import LlmAgent

tools = load_adk_tools(Path(".pretensor"))
agent = LlmAgent(model="gemini-2.0-flash", tools=tools)
```

---

## Tool reference

| Tool | Description | Required args |
|---|---|---|
| `schema` | Node labels, edge types, and properties | `database` |
| `context` | Full table context: columns, relationships, lineage, cluster | `table` |
| `traverse` | Join path between two tables with SQL hints | `from_table`, `to_table`, `database` |
| `impact` | Downstream tables reachable from a table (by hop depth) | `table`, `database` |
| `query` | BM25 keyword search over table and entity metadata | `q` |
| `validate_sql` | Validate SQL against indexed graph (unknown tables/columns, invalid joins) | `sql`, `database` |

All tools return the same `dict` structure as the MCP server tools.
