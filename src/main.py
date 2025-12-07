from src.services.wikipedia_service import WikipediaAPI
from src.agent.page_visitor import PageVisitor


if __name__ == "__main__":
    api = WikipediaAPI(
        page_request_limit=6500,
        wikirank_datasets_with_quality_scores_en_tsv="en.tsv"
    )
    visitor = PageVisitor(api)
    visitor.collect_first_pages()
    while api.get_usage_summary()["page_requests_used"] < api.get_usage_summary()["page_request_limit"]:
        print(visitor.requests_limit, "Get next page")
        page = visitor.get_next_page()
        page.retrieve(api)
        if page.content:
            print("Page retrieved:", page.name)
            visitor.process_new_page_content(page)
    top_pages = visitor.find_top_pages()
    visitor.save_pages(top_pages)