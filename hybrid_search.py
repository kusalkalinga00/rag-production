from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


documents = [
    Document(
        page_content="SKU: UWP-1042. The UltraWidget Pro is a high-performance widget designed for enterprise use. It supports up to 10,000 operations per second and integrates with all major platforms.",
        metadata={"type": "product", "sku": "UWP-1042"},
    ),
    Document(
        page_content="SKU: MWL-0231. The MiniWidget Lite is a compact, budget-friendly widget suitable for small businesses. It supports basic operations and is compatible with standard platforms.",
        metadata={"type": "product", "sku": "MWL-0231"},
    ),
    Document(
        page_content="If the application fails to start, check that all environment variables are set correctly. Restart the service and verify that the database connection string is valid.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="When the dashboard does not load, clear the browser cache, disable extensions, and ensure the API gateway is reachable from the client network.",
        metadata={"type": "troubleshooting"},
    ),
    Document(
        page_content="Error E4023: Database connection timeout. This error occurs when the database server is unreachable. Verify network connectivity and firewall rules.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="Error E5011: Invalid API key. The provided API key is expired or revoked. Generate a new key from the developer portal and update your configuration.",
        metadata={"type": "error"},
    ),
    Document(
        page_content="Authentication is handled via OAuth 2.0. Users must obtain an access token by submitting their credentials to the /auth/token endpoint. Tokens expire after 3600 seconds.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="Multi-factor authentication (MFA) can be enabled in the security settings. Supported methods include TOTP authenticator apps and SMS-based one-time passwords.",
        metadata={"type": "auth"},
    ),
    Document(
        page_content="The application reads its configuration from a .env file or environment variables. Required keys include DATABASE_URL, API_KEY, and LOG_LEVEL.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="Feature flags can be toggled in the config.yaml file under the features section. Set the value to true to enable a feature and false to disable it.",
        metadata={"type": "config"},
    ),
    Document(
        page_content="All user data is stored in compliance with GDPR regulations. Personal data is encrypted at rest and in transit. Users may request data deletion via the privacy portal.",
        metadata={"type": "compliance"},
    ),
    Document(
        page_content="The platform is SOC 2 Type II certified. Security audits are conducted annually. Access logs are retained for a minimum of 12 months.",
        metadata={"type": "compliance"},
    ),
]

print(f"Loaded {len(documents)} documents into memory.")


vector_store = Chroma.from_documents(
    documents,
    embeddings,
    collection_name="hybrid_test_collection",
)

vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("Vector Retriever initialized with 3 nearest neighbors.")

bm25_retriever = BM25Retriever.from_documents(documents, k=3)
print("BM25 Retriever initialized with 3 nearest neighbors.")


hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],
)

print("Hybrid Retriever initialized with equal weights for vector and BM25 retrievers.")


def test_hybrid_search():
    """
    Tests the hybrid retriever across queries designed to exercise different
    retrieval strengths:

    - Exact keyword / BM25 strength: SKU lookup, error codes
    - Semantic / vector strength: paraphrased intent, conceptual questions
    - Overlap queries: both retrievers should contribute
    - Edge cases: short single-word queries, multi-concept queries
    """
    test_cases = [
        # --- Exact keyword match (BM25 should dominate) ---
        {
            "label": "SKU exact lookup",
            "query": "UWP-1042",
            "expect_type": "product",
        },
        {
            "label": "Error code exact lookup",
            "query": "E4023",
            "expect_type": "error",
        },
        # --- Semantic / paraphrase (vector should dominate) ---
        {
            "label": "Semantic - product capability question",
            "query": "What widget is best suited for large organisations with high throughput needs?",
            "expect_type": "product",
        },
        {
            "label": "Semantic - troubleshooting paraphrase",
            "query": "The app won't launch after deployment, how do I fix it?",
            "expect_type": "troubleshooting",
        },
        {
            "label": "Semantic - auth concept",
            "query": "How do I log in securely using a token-based system?",
            "expect_type": "auth",
        },
        # --- Keyword + semantic overlap (both retrievers contribute) ---
        {
            "label": "Error with natural language context",
            "query": "database connection timeout error",
            "expect_type": "error",
        },
        {
            "label": "Config keyword + concept",
            "query": "environment variables configuration file",
            "expect_type": "config",
        },
        {
            "label": "Compliance keyword + concept",
            "query": "GDPR data privacy user rights",
            "expect_type": "compliance",
        },
        # --- Short / single-word queries (stress test for BM25 sparsity) ---
        {
            "label": "Single keyword - auth",
            "query": "OAuth",
            "expect_type": "auth",
        },
        {
            "label": "Single keyword - compliance",
            "query": "SOC",
            "expect_type": "compliance",
        },
        # --- Multi-concept query (tests ranking fusion) ---
        {
            "label": "Multi-concept - MFA and security settings",
            "query": "enable two-factor authentication in security settings",
            "expect_type": "auth",
        },
        {
            "label": "Multi-concept - feature flags and config",
            "query": "toggle features in YAML configuration",
            "expect_type": "config",
        },
    ]

    print("\n" + "=" * 70)
    print("HYBRID SEARCH TEST RESULTS")
    print("=" * 70)

    passed = 0
    for tc in test_cases:
        results = hybrid_retriever.invoke(tc["query"])
        top_types = [doc.metadata.get("type", "unknown") for doc in results]
        hit = tc["expect_type"] in top_types
        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1

        print(f"\n[{status}] {tc['label']}")
        print(f"  Query       : {tc['query']}")
        print(f"  Expect type : {tc['expect_type']}")
        print(f"  Got types   : {top_types}")
        for i, doc in enumerate(results, 1):
            snippet = doc.page_content[:80].replace("\n", " ")
            print(f"    {i}. [{doc.metadata.get('type')}] {snippet}...")

    print("\n" + "=" * 70)
    print(f"Result: {passed}/{len(test_cases)} tests passed.")
    print("=" * 70 + "\n")


test_hybrid_search()
