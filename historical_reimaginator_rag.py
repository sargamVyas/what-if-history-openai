# Historical Reimaginator with RAG
# Uses Wikipedia → Pinecone (vector store) + OpenAI embeddings
# Historian Agent retrieves relevant context via RAG before web search

import os
import time
import wikipedia
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, BaseTool
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# ── Environment ──────────────────────────────────────────────────────────────
load_dotenv()

openai_api_key  = os.getenv("OPENAI_API_KEY")
serper_api_key  = os.getenv("SERPER_API_KEY")
pinecone_api_key   = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "historical-rag")

for name, val in [
    ("OpenAI API key",   openai_api_key),
    ("Serper API key",   serper_api_key),
    ("Pinecone API key", pinecone_api_key),
]:
    if not val:
        st.error(f"{name} is missing. Please add it to the .env file.")
        st.stop()

# ── OpenAI client ─────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=openai_api_key)

# ── Pinecone setup ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 50

pc = Pinecone(api_key=pinecone_api_key)

def get_or_create_index(index_name: str):
    """Return a Pinecone index, creating it if needed."""
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Wait until ready
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    return pc.Index(index_name)

# ── Helper functions ───────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings with OpenAI."""
    response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def ingest_wikipedia(topic: str, index) -> int:
    """
    Fetch Wikipedia content for *topic*, chunk it, embed it, and upsert
    into Pinecone under a namespace derived from the topic.
    Returns the number of chunks ingested.
    """
    namespace = topic.lower().replace(" ", "_")[:60]

    # Skip re-ingestion if namespace already has vectors
    stats = index.describe_index_stats()
    if namespace in stats.get("namespaces", {}) and \
       stats["namespaces"][namespace].get("vector_count", 0) > 0:
        return stats["namespaces"][namespace]["vector_count"]

    # Wikipedia search + full page fetch
    try:
        search_results = wikipedia.search(topic, results=3)
        pages_text = []
        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                pages_text.append(f"# {page.title}\n{page.content}")
            except (wikipedia.DisambiguationError, wikipedia.PageError):
                continue
    except Exception as e:
        st.warning(f"Wikipedia fetch warning: {e}")
        pages_text = []

    if not pages_text:
        return 0

    full_text = "\n\n".join(pages_text)
    chunks = chunk_text(full_text)

    # Embed in batches of 50
    batch_size = 50
    vectors = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embeddings = embed(batch)
        for j, (chunk, emb) in enumerate(zip(batch, embeddings)):
            vectors.append({
                "id":       f"{namespace}_{i+j}",
                "values":   emb,
                "metadata": {"text": chunk, "topic": topic},
            })

    # Upsert to Pinecone
    upsert_batch = 100
    for i in range(0, len(vectors), upsert_batch):
        index.upsert(vectors=vectors[i : i + upsert_batch], namespace=namespace)

    return len(vectors)


def retrieve(query: str, topic: str, index, top_k: int = 5) -> str:
    """Embed *query* and return the top_k most relevant chunks as a string."""
    namespace = topic.lower().replace(" ", "_")[:60]
    query_emb  = embed([query])[0]
    results    = index.query(
        vector=query_emb,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
    )
    if not results["matches"]:
        return "No relevant context found in the knowledge base."

    passages = []
    for i, match in enumerate(results["matches"], 1):
        score = match["score"]
        text  = match["metadata"].get("text", "")
        passages.append(f"[Passage {i} | similarity={score:.3f}]\n{text}")
    return "\n\n---\n\n".join(passages)

# ── Custom CrewAI RAG Tool ────────────────────────────────────────────────────

class HistoricalRAGTool(BaseTool):
    """
    Retrieves relevant historical context from the Pinecone vector store.
    Call this tool with a specific question or keyword before using web search.
    """
    name: str = "Historical RAG Retriever"
    description: str = (
        "Retrieves relevant historical context from a pre-built Wikipedia knowledge base "
        "stored in Pinecone. Input should be a specific historical question or keyword. "
        "Always use this tool FIRST before searching the web."
    )
    index: object = None      # Pinecone Index instance (injected at runtime)
    topic: str = ""           # Current event topic (injected at runtime)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        if self.index is None:
            return "RAG tool not initialised — no index available."
        return retrieve(query, self.topic, self.index)

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Reimagined Historical Event", layout="wide")
st.title("Reimagined Historical Event")
st.markdown(
    "Generate a comprehensive blog post about how the world would look "
    "if a historical event had **not** happened — now powered by RAG."
)

with st.sidebar:
    st.header("Content Settings")

    topic = st.text_area(
        "Enter historical event",
        height=100,
        placeholder="e.g. World War II, The French Revolution, The Moon Landing",
    )

    st.markdown("Temperature Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)

    st.markdown("RAG Settings")
    top_k = st.slider("Top-K retrieved passages", 3, 10, 5)

    st.markdown("---")
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    with st.expander("How to use"):
        st.markdown("""
        1. Enter a historical event in the text area above
        2. Adjust temperature (higher = more creative)
        3. Click **Generate Content**
        4. The app will:
           - Fetch Wikipedia articles on your topic
           - Embed & store them in Pinecone
           - Run three AI agents (Historian → Alternate Reality → Storyteller)
           - The Historian uses RAG *before* web search for grounded research
        5. Download the result as a markdown file
        """)

# ── Core generation function ───────────────────────────────────────────────────

def generate_content(topic: str, temperature: float, top_k: int):

    # 1. Prepare vector store
    with st.status("🗄️ Setting up Pinecone index…"):
        index = get_or_create_index(pinecone_index_name)

    # 2. Ingest Wikipedia
    with st.status(f"📚 Ingesting Wikipedia articles for '{topic}'…"):
        n_chunks = ingest_wikipedia(topic, index)
        st.write(f"✅ {n_chunks} chunks available in Pinecone.")

    # 3. Build tools
    rag_tool    = HistoricalRAGTool(index=index, topic=topic)
    search_tool = SerperDevTool(api_key=serper_api_key, n_results=10)

    llm = LLM(model="gpt-3.5-turbo", temperature=temperature, api_key=openai_api_key)

    # ── Agents ────────────────────────────────────────────────────────────────
    historian_agent = Agent(
        role="Historian Agent",
        goal=f"Research and provide factual, well-documented historical accounts related to {topic}.",
        backstory=(
            "You're a highly skilled historian with deep knowledge of world history, "
            "key events, and their impacts. You specialize in fact-based research, "
            "providing verified historical information with citations. "
            "You ALWAYS start by querying the Historical RAG Retriever tool to retrieve "
            "pre-fetched Wikipedia context before searching the web."
        ),
        allow_delegation=False,
        verbose=True,
        tools=[rag_tool, search_tool],
        llm=llm,
    )

    alternate_reality_agent = Agent(
        role="Alternate Reality Generator Agent",
        goal=f"Analyse and generate plausible alternate history scenarios based on the modification of {topic}.",
        backstory=(
            "You're an expert in speculative history and counterfactual analysis. "
            "You excel at logical reasoning, identifying key turning points in history, "
            "and predicting the cascading effects of alternative outcomes."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    storyteller_agent = Agent(
        role="Storyteller Agent",
        goal=f"Craft immersive and engaging alternative history narratives based on the scenario of {topic}.",
        backstory=(
            "You're a talented writer specialising in historical fiction and alternate history. "
            "You translate complex historical and speculative ideas into compelling narratives."
        ),
        allow_delegation=False,
        verbose=True,
        llm=llm,
    )

    # ── Tasks ──────────────────────────────────────────────────────────────────
    historical_research_task = Task(
        description=(
            f"Research the historical event: **{topic}**.\n\n"
            "Steps:\n"
            "1. FIRST, use the 'Historical RAG Retriever' tool with relevant queries "
            "   to pull pre-ingested Wikipedia context from the knowledge base.\n"
            "2. THEN, use web search to supplement and verify findings.\n"
            "3. Compile a structured research brief covering:\n"
            "   - Key events, dates, and figures\n"
            "   - Political, social, and economic impacts\n"
            "   - Verified sources and citations\n"
            "   - A timeline of significant developments"
        ),
        expected_output=(
            "A structured historical research brief containing:\n"
            "- A concise summary of the event with key details\n"
            "- Analysed impacts on politics, society, and technology\n"
            "- Verified sources with citations and references\n"
            "- Timeline of significant developments\n"
            "- Key insights in bullet-point form"
        ),
        agent=historian_agent,
    )

    alternate_history_task = Task(
        description=(
            "Based on the historical research brief, generate a plausible alternate history:\n"
            "1. Identify key decision points where history could have diverged\n"
            "2. Explore logical consequences: political, economic, technological, social\n"
            "3. Project long-term ripple effects\n"
            "4. Ensure internal consistency with known historical dynamics"
        ),
        expected_output=(
            "A well-structured alternate history analysis containing:\n"
            "- Description of the altered event and why it changed\n"
            "- Cause-and-effect analysis of immediate consequences\n"
            "- Long-term projection of global impact\n"
            "- Multiple potential outcomes\n"
            "- Final comparison: reality vs. alternate timeline"
        ),
        agent=alternate_reality_agent,
    )

    storytelling_task = Task(
        description=(
            "Using the alternate history scenario, craft an immersive narrative:\n"
            "1. Write in historical fiction / documentary style\n"
            "2. Include key characters and their roles in the altered world\n"
            "3. Build dramatic tension with a clear beginning, middle, and end\n"
            "4. Conclude with thought-provoking insights\n"
            "5. Format in markdown with clear sections and headings"
        ),
        expected_output=(
            "A captivating alternate history story formatted in markdown that:\n"
            "- Reads like a historical fiction piece or documentary script\n"
            "- Uses vivid descriptions and character perspectives\n"
            "- Reflects realistic consequences of the alternate timeline"
        ),
        agent=storyteller_agent,
    )

    # ── Crew ───────────────────────────────────────────────────────────────────
    crew = Crew(
        agents=[historian_agent, alternate_reality_agent, storyteller_agent],
        tasks=[historical_research_task, alternate_history_task, storytelling_task],
        verbose=True,
    )

    return crew.kickoff(inputs={"topic": topic})


# ── Main ───────────────────────────────────────────────────────────────────────
if generate_button:
    if not topic.strip():
        st.warning("Please enter a historical event before generating.")
    else:
        with st.spinner("Generating content… this may take a few minutes."):
            try:
                result = generate_content(topic, temperature, top_k)
                st.markdown("### Generated Content")
                st.markdown(result)

                st.download_button(
                    label="Download Content",
                    data=result.raw,
                    file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

st.markdown("---")
