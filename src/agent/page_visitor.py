from time import time
import faiss
from src.domain.page import Page


class PageVisitor:
    startkit = [
        # ── Fundamental sciences ───────────────────────────────
        "Mathematics", "Physics", "Chemistry", "Biology", "Astronomy", "Geology", "Computer science",
        "Quantum mechanics", "General relativity", "Thermodynamics", "Electromagnetism", "Organic chemistry",
        "Molecular biology", "Genetics", "Neuroscience", "Astrophysics", "Particle physics", "Algebra",
        "Calculus", "Topology", "Number theory", "Graph theory", "Biochemistry", "Evolutionary biology",

        # ── Technology and computer science ─────────────────────
        "Artificial intelligence", "Machine learning", "Deep learning", "Cryptography", "Blockchain",
        "Quantum computing", "Robotics", "Cybersecurity", "Computer vision", "Natural language processing",
        "Operating system", "Programming language", "Internet", "Virtual reality", "Augmented reality",
        "Internet of Things", "3D printing", "Nanotechnology", "Renewable energy", "Autonomous vehicles",

        # ── History and civilizations ───────────────────────────
        "Ancient Egypt", "Ancient Greece", "Ancient Rome", "Ancient China", "Mesopotamia", "Mayan civilization",
        "Inca Empire", "Aztec civilization", "Byzantine Empire", "Ottoman Empire", "Mongol Empire",
        "Renaissance", "Middle Ages", "Industrial Revolution", "French Revolution", "World War I",
        "World War II", "Cold War", "American Civil War", "Russian Revolution", "Viking Age",

        # ── Art and Architecture ────────────────────────────────
        "Renaissance art", "Impressionism", "Surrealism", "Cubism", "Abstract art", "Baroque", "Rococo",
        "Modern architecture", "Gothic architecture", "Islamic architecture", "Ancient Greek architecture",
        "Romanesque architecture", "Art Deco", "Street art", "Graffiti", "Sculpture", "Photography",

        # ── Music and genres ────────────────────────────────────
        "Classical music", "Opera", "Jazz", "Rock music", "Hip hop", "Electronic music", "Folk music",
        "Baroque music", "Romantic music", "Blues", "Reggae", "Punk rock", "Symphony", "Chamber music",

        # ── Literature and theater ──────────────────────────────
        "William Shakespeare", "Greek tragedy", "Epic poetry", "Science fiction", "Fantasy literature",
        "Modernist literature", "Victorian literature", "Russian literature", "Japanese literature",
        "Latin American literature", "Postmodern literature", "Theatre", "Drama", "Comedy",

        # ── Philosophy and religion ─────────────────────────────
        "Philosophy", "Ethics", "Metaphysics", "Epistemology", "Political philosophy", "Existentialism",
        "Stoicism", "Buddhism", "Hinduism", "Christianity", "Islam", "Judaism", "Confucianism", "Taoism",

        # ── Mythology and folklore ──────────────────────────────
        "Greek mythology", "Norse mythology", "Egyptian mythology", "Roman mythology", "Hindu mythology",
        "Celtic mythology", "Japanese mythology", "Slavic mythology", "Native American mythology",

        # ── Geography and nature ────────────────────────────────
        "country", "capital city", "river", "mountain", "ocean", "desert", "volcano", "earthquake",
        "climate change", "ecology", "biodiversity", "coral reef", "rainforest", "glacier",

        # ── Chemistry and Biology (elements and species) ────────
        "chemical element", "mineral", "dinosaur", "extinct animal", "bird", "mammal", "fish", "insect",

        # ── Space and astronomy ─────────────────────────────────
        "planet", "star", "galaxy", "black hole", "constellation", "space exploration", "telescope",

        # ── Medicine and health ─────────────────────────────────
        "Medicine", "human disease", "virus", "vaccine", "cancer", "genetic disorder", "neuroscience",

        # ── Sport and games ─────────────────────────────────────
        "Olympic Games", "Football", "Cricket", "Tennis", "Chess", "board game", "card game",

        # ── Economics, politics, society ────────────────────────
        "Economics", "Democracy", "Capitalism", "Socialism", "Law", "Human rights", "Feminism",
        "Psychology", "Sociology", "Anthropology", "Linguistics", "Archaeology", "Ethnography",

        # ── Nobel laureates and great personalities ─────────────
        "Nobel Prize", "mathematician", "physicist", "writer", "composer", "painter", "filmmaker",

        # ── Wide "golden" queries ───────────────────────────────
        "war", "treaty", "revolution", "empire", "kingdom", "philosopher", "mathematical theorem",
        "physical constant", "literary genre", "film genre", "music genre", "architectural style",
        "art movement", "historical period", "ancient civilization", "world religion",

        # ── Most significant single articles ────────────────────
        "Earth", "Universe", "Human", "Life", "Time", "Energy", "Evolution", "Civilization",
        "Language", "Consciousness", "Internet", "Democracy", "Capitalism", "World War"
    ]

    def __init__(self, api, requests_limit: int = 6500) -> None:
        self.api = api
        self.requests_limit = requests_limit
        self.available_pages: list[Page] = []
        self.retreived_pages: dict[str, Page] = {}
        self.link_count: dict[str, int] = {}

        self.index = faiss.IndexFlatIP(384)
        self.page_weights_cache: dict[Page, float] = {}

    def collect_first_pages(self) -> None:
        """Run the starting search querries to collect the first available pages."""
        for search_request in self.startkit:
            print(search_request)
            for page_name in self.api.search_pages(search_request):
                self._process_new_available_page(page_name, 0)
            self.requests_limit -= 1

    def _process_new_available_page(self, page_name: str, depth: int) -> None:
        """Add the new available page or increase its link count."""
        if page_name in self.link_count:
            self.link_count[page_name] += 1
            return
        if not self._filter_page(page_name):
            return
        page = Page(name=page_name, depth=depth)
        if self.index.ntotal > 0:
            distances, _ = self.index.search(
                page.name_emb_normalized.reshape(1, -1),
                k=1
            )
            max_similarity = distances[0][0]
            page.novelty = max(0.0, 1.0 - max_similarity)
        else:
            page.novelty = 1.0

        self.index.add(page.name_emb_normalized.reshape(1, -1))

        self.available_pages.append(page)
        self.link_count[page_name] = 1
        self.page_weights_cache[page] = self._compute_weight(page)

    def _compute_weight(self, page: Page) -> float:
        """Weight for choosing the next page"""
        return page.get_name_rank() * page.novelty * (1 / (1 + self.link_count[page.name]))

    def _filter_page(self, page_name: str) -> bool:
        """Identify whether the page with the provided name is suitable for us."""
        if page_name in self.retreived_pages:
            return False
        if not self.api.is_legal_page(page_name):
            return False
        if "disambiguation" in page_name.lower() or "outline" in page_name.lower():
            return False
        if page_name.startswith("List of"):
            return False

        return True

    def get_next_page(self) -> Page:
        """Find the next page to retrieve."""
        page = max(self.available_pages, key=lambda p: self.page_weights_cache[p])
        self.available_pages.remove(page)
        self.page_weights_cache.pop(page)
        self.link_count.pop(page.name)
        return page

    def process_new_page_content(self, page: Page) -> None:
        """Process the retrieved page."""
        if not page.content:
            return
        self.requests_limit -= 1
        self.retreived_pages[page.name] = page
        if page.depth == 3:
            return
        start = time()
        for link_page_name in page.content["links"]:
            self._process_new_available_page(link_page_name, page.depth + 1)
        print(time() - start)

    def find_top_pages(self, n: int = 5000) -> list[str]:
        """Identify top 5000 pages which are the most suitable for us."""
        pages = self.retreived_pages.items()
        pages = sorted(pages, key=lambda x: x[1].get_rank(), reverse=True)
        pages = [page.content['title'] for _name, page in pages if page.content]
        if len(pages) > n:
            return pages[:n]
        else:
            return pages

    def save_pages(self, top_pages: list[str]) -> None:
        """Save the most suitable pages to the files."""
        for page_name in top_pages:
            self.api.save_page(page_name)
        from ..services.dataset_service import DatasetService
        dataset_service = DatasetService(self.api)
        dataset_service.save_dataset("test.pkl", "test.csv")