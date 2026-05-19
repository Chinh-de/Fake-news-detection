import sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

from retriever import search_web_evidence, VI_NEWS_DOMAINS

query = "Chinh phu hop phien thuong ky thang 5 GDP 2025"
print("Testing Web RAG for query:", query[:50])
print("Domains:", VI_NEWS_DOMAINS[:3], "...")
print()

results = search_web_evidence(query, top_k=3, crawl=False)
print(f"Web results: {len(results)}")
for r in results:
    rk = r["rank"]
    dom = r["domain"]
    title = r["title"][:50]
    sc = r["score"]
    url = r["url"][:60]
    print(f"  #{rk} [{dom}] {title}")
    print(f"       score={sc:.3f} | {url}")
