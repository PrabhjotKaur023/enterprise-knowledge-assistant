"""
Agentic AI layer using LangChain.

Three agents:
- DocumentAgent: answers questions by searching uploaded documents (RAG-based)
- SQLAgent: answers structured questions by querying the metadata database
- SummaryAgent: generates document summaries

I went back and forth on whether to use LangChain agents or build my own
tool-calling loop. LangChain is fine for standard use cases but the abstraction
gets frustrating when you need to debug what the agent is doing internally.
For now it's good enough.
"""

import logging
from typing import Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase

from app.core.config import settings
from app.pipeline.rag_engine import get_rag_engine

logger = logging.getLogger(__name__)


def _get_langchain_llm():
    """Return a LangChain-compatible LLM based on configured provider."""
    if settings.LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
    elif settings.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL or "gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")


class DocumentAgent:
    """
    Agent that uses RAG to answer questions about uploaded documents.
    Wraps the RAG engine as a LangChain tool so the agent can decide
    when to invoke document search vs answer from context.
    """

    def __init__(self):
        self.rag_engine = get_rag_engine()
        self._executor: Optional[AgentExecutor] = None

    def _build_executor(self) -> AgentExecutor:
        llm = _get_langchain_llm()

        def document_search(query: str) -> str:
            result = self.rag_engine.answer(query)
            if not result.sources:
                return "No relevant information found in uploaded documents."
            sources_text = "\n".join(
                f"- [{s.filename}]: {s.content[:200]}..." for s in result.sources
            )
            return f"Answer: {result.answer}\n\nSources:\n{sources_text}"

        tools = [
            Tool(
                name="document_search",
                func=document_search,
                description=(
                    "Search the uploaded documents for information. "
                    "Use this for any question about document content. "
                    "Input: the search query."
                ),
            )
        ]

        prompt = PromptTemplate.from_template(
            """You are a helpful document assistant. Use the available tools to answer questions.

Available tools:
{tools}

Tool names: {tool_names}

To use a tool, format your response as:
Thought: [your reasoning]
Action: [tool name]
Action Input: [input to the tool]
Observation: [tool result]
... (repeat as needed)
Thought: I now have enough information to answer.
Final Answer: [your final response]

Question: {input}
{agent_scratchpad}"""
        )

        agent = create_react_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=4,
            verbose=settings.DEBUG,
            handle_parsing_errors=True,
        )

    def run(self, query: str) -> str:
        if self._executor is None:
            self._executor = self._build_executor()
        try:
            result = self._executor.invoke({"input": query})
            return result.get("output", "No response generated.")
        except Exception as e:
            logger.error(f"DocumentAgent failed: {e}")
            # Fall back to direct RAG call
            fallback = self.rag_engine.answer(query)
            return fallback.answer


class SQLAgent:
    """
    Agent that can query the document metadata database to answer
    structured questions like "how many documents have been uploaded?"
    or "what files did I upload last week?".

    Note: this only has READ access to non-sensitive tables (documents, sessions).
    Chat message content is excluded for privacy.
    """

    def __init__(self):
        self._executor: Optional[AgentExecutor] = None

    def _build_executor(self) -> AgentExecutor:
        from langchain_community.agent_toolkits import create_sql_agent

        # Only expose metadata tables — not chat content
        db = SQLDatabase.from_uri(
            settings.DATABASE_URL,
            include_tables=["documents", "chat_sessions"],
        )

        llm = _get_langchain_llm()
        return create_sql_agent(
            llm=llm,
            db=db,
            agent_type="openai-tools" if settings.LLM_PROVIDER == "openai" else "zero-shot-react-description",
            verbose=settings.DEBUG,
            max_iterations=5,
        )

    def run(self, query: str) -> str:
        if self._executor is None:
            self._executor = self._build_executor()
        try:
            result = self._executor.invoke({"input": query})
            return result.get("output", "No response generated.")
        except Exception as e:
            logger.error(f"SQLAgent failed: {e}")
            return f"I couldn't answer that question via the database. Error: {str(e)}"


class SummaryAgent:
    """
    Generates document summaries. Simple enough that it doesn't need
    a full ReAct agent — just a direct LLM call through the RAG engine.
    """

    def __init__(self):
        self.rag_engine = get_rag_engine()

    def summarize(self, document_id: str, filename: str) -> str:
        logger.info(f"Generating summary for document: {filename}")
        return self.rag_engine.summarize_document(document_id, filename)


class AgentRouter:
    """
    Decides which agent to use based on the query type.
    Simple keyword matching for now — could use a classifier later.

    TODO: replace with intent classification using embeddings
    """

    SQL_KEYWORDS = {
        "how many", "count", "list all", "show me all", "uploaded",
        "total", "when did", "last week", "yesterday", "recent uploads"
    }

    def __init__(self):
        self.document_agent = DocumentAgent()
        self.sql_agent = SQLAgent()
        self.summary_agent = SummaryAgent()

    def route_and_run(self, query: str, document_id: Optional[str] = None) -> str:
        """Route the query to the appropriate agent."""
        query_lower = query.lower()

        if document_id and any(w in query_lower for w in ["summarize", "summary", "overview", "brief"]):
            logger.info("Routing to SummaryAgent")
            return self.summary_agent.summarize(document_id, "document")

        if any(keyword in query_lower for keyword in self.SQL_KEYWORDS):
            logger.info("Routing to SQLAgent")
            return self.sql_agent.run(query)

        logger.info("Routing to DocumentAgent")
        return self.document_agent.run(query)
