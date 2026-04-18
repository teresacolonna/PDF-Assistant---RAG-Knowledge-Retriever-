from dataclasses import dataclass
from src.rag_pipeline import answer_question, answer_question_with_graph
from src.vector_store import get_vector_store
from src.retriever import retrieve_context

@dataclass
class TestCase:
    question: str
    expected_keywords: list[str]

TEST_CASES = [
    TestCase("Come sono cambiate le emissioni di CO2 tra 2020 e 2023?", ["emissioni", "2020", "2023"]),
    TestCase("Qual e il trend del PIL verde nei tre anni?", ["pil", "verde", "trend"]),
    TestCase("In quale anno il consumo energetico e stato piu alto?", ["consumo", "energetico", "anno"]),
]

def run_tests(k: int = 3, year_filter: str | None = None):
    vs = get_vector_store(rebuild=False)
    for tc in TEST_CASES:
        print("Q:", tc.question)
        retrieved, context = retrieve_context(vs, tc.question, k=k)
        print("  retrieval count:", len(retrieved))
        for doc, score in retrieved:
            print(f"   - score {score} | fonte {doc.metadata.get('fonte')} | pagina {doc.metadata.get('pagina')}")
        ans = answer_question(tc.question, k=k, year_filter=year_filter)
        print("A:", ans)
        seen = {kw: kw in ans.lower() for kw in tc.expected_keywords}
        print("Keywords presenti:", seen)
        ans2 = answer_question_with_graph(tc.question, k=k, year_filter=year_filter)
        print("A (con grafo):", ans2)
        seen2 = {kw: kw in ans2.lower() for kw in tc.expected_keywords}
        print("Keywords presenti (grafo):", seen2)
        print("-" * 60)

if __name__ == "__main__":
    run_tests()
