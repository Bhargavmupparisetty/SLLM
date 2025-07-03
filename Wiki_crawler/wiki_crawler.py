import requests
from bs4 import BeautifulSoup
import os
import time
import json
import logging
import hashlib
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse, quote
from typing import Dict, List, Optional, Tuple
import random
from dataclasses import dataclass, asdict
import pickle
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import sys

@dataclass
class GeneralArticleData:
    
    article_id: str
    title: str
    url: str
    content: str
    category: str = ""
    summary: str = ""
    content_hash: str = ""
    scraped_at: str = ""
    word_count: int = 0
    language: str = "en"
    domain: str = ""

class WikipediaSpecializedCrawler:
    def __init__(self, config_file: str = "wiki_specialized_crawler_config.json"):
        self.config = self._load_config(config_file)
        self.setup_logging()
        self.setup_directories()
        self.visited_urls = set()
        self.scraped_hashes = set()
        self.failed_urls = {}
        self.article_count = 0
        self.session = self._create_session()
        self.progress_file = os.path.join(self.config['save_dir'], 'wiki_specialized_crawler_progress.pkl')
        self.running = True
        self.lock = threading.Lock()
        
        self.output_file = os.path.join(self.config['save_dir'], 'specialized_knowledge_training_data.txt')
        
        # Wikipedia API endpoints
        self.wiki_api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wiki_search_url = "https://en.wikipedia.org/w/api.php"
        
        self._load_progress()
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self, config_file: str) -> Dict:

        search_batches = [
            # Quantum Physics & Quantum Mechanics
            ["quantum mechanics", "quantum field theory", "quantum entanglement", "quantum superposition", "quantum decoherence"],
            ["quantum tunneling", "quantum chromodynamics", "quantum electrodynamics", "quantum gravity", "quantum computing"],
            ["wave function", "Schrödinger equation", "Heisenberg uncertainty principle", "quantum state", "quantum measurement"],
            ["Bell's theorem", "EPR paradox", "quantum teleportation", "quantum cryptography", "quantum annealing"],
            ["many-worlds interpretation", "Copenhagen interpretation", "pilot wave theory", "quantum foam", "vacuum energy"],
            ["quantum phase transition", "quantum spin", "quantum dot", "quantum well", "quantum oscillator"],
            ["Bose-Einstein condensate", "quantum statistics", "fermions", "bosons", "Pauli exclusion principle"],
            ["quantum interference", "double-slit experiment", "quantum eraser", "delayed choice experiment", "quantum Zeno effect"],
            
            # Mathematics Theorems & Pure Mathematics
            ["Fermat's last theorem", "Gödel's incompleteness theorems", "four color theorem", "prime number theorem", "Riemann hypothesis"],
            ["Poincaré conjecture", "Millennium Prize Problems", "P versus NP problem", "Goldbach conjecture", "twin prime conjecture"],
            ["Pythagorean theorem", "fundamental theorem of calculus", "fundamental theorem of algebra", "Bayes' theorem", "central limit theorem"],
            ["Green's theorem", "Stokes' theorem", "divergence theorem", "intermediate value theorem", "mean value theorem"],
            ["Nash equilibrium", "fixed point theorem", "Brouwer fixed point theorem", "Banach fixed point theorem", "Kakutani fixed point theorem"],
            ["category theory", "group theory", "ring theory", "field theory", "Galois theory"],
            ["topology", "differential geometry", "algebraic geometry", "number theory", "combinatorics"],
            ["graph theory", "Ramsey theory", "extremal graph theory", "chromatic polynomial", "planar graph"],
            ["mathematical logic", "set theory", "model theory", "proof theory", "recursion theory"],
            ["functional analysis", "measure theory", "real analysis", "complex analysis", "harmonic analysis"],
            
            # Advanced Biology & Molecular Biology
            ["DNA replication", "transcription", "translation", "protein folding", "enzyme kinetics"],
            ["cell cycle", "mitosis", "meiosis", "apoptosis", "autophagy"],
            ["photosynthesis", "cellular respiration", "ATP synthesis", "electron transport chain", "Calvin cycle"],
            ["gene expression", "gene regulation", "epigenetics", "chromatin remodeling", "histone modification"],
            ["signal transduction", "receptor proteins", "G-protein coupled receptors", "tyrosine kinase", "phosphorylation"],
            ["membrane transport", "ion channels", "active transport", "passive transport", "endocytosis"],
            ["immune system", "adaptive immunity", "innate immunity", "antibodies", "T cells"],
            ["neurobiology", "action potential", "synaptic transmission", "neurotransmitters", "neuroplasticity"],
            ["evolutionary biology", "natural selection", "genetic drift", "speciation", "phylogenetics"],
            ["developmental biology", "embryogenesis", "morphogenesis", "cell differentiation", "stem cells"],
            ["molecular evolution", "horizontal gene transfer", "codon usage bias", "molecular clock", "comparative genomics"],
            ["biochemical pathways", "metabolic networks", "glycolysis", "citric acid cycle", "pentose phosphate pathway"],
            
            # Quantum Biology & Biophysics
            ["quantum biology", "quantum coherence in biology", "quantum effects in photosynthesis", "quantum tunneling in enzymes", "magnetoreception"],
            ["protein dynamics", "molecular motors", "DNA mechanics", "membrane biophysics", "single molecule biophysics"],
            
            # Mathematical Biology & Computational Biology
            ["population dynamics", "predator-prey models", "mathematical epidemiology", "reaction-diffusion equations", "pattern formation"],
            ["bioinformatics", "computational biology", "systems biology", "network biology", "mathematical modeling"],
            
            # Advanced Physics & Mathematical Physics
            ["string theory", "supersymmetry", "extra dimensions", "gauge theory", "Yang-Mills theory"],
            ["general relativity", "special relativity", "black holes", "gravitational waves", "cosmology"],
            ["statistical mechanics", "thermodynamics", "phase transitions", "critical phenomena", "renormalization"],
            ["particle physics", "standard model", "Higgs mechanism", "quantum field theory", "lattice QCD"]
        ]

        default_config = {
            "base_url": "https://en.wikipedia.org",
            "specialized_categories": [
                # Quantum Physics
                "Quantum_mechanics", "Quantum_field_theory", "Quantum_computing", "Quantum_information",
                "Quantum_optics", "Quantum_chemistry", "Quantum_biology",
                
                # Mathematics
                "Mathematical_theorems", "Algebra", "Analysis", "Topology", "Geometry", "Number_theory",
                "Combinatorics", "Graph_theory", "Mathematical_logic", "Set_theory", "Category_theory",
                "Differential_equations", "Probability_theory", "Statistics", "Mathematical_physics",
                
                # Biology
                "Molecular_biology", "Cell_biology", "Genetics", "Biochemistry", "Developmental_biology",
                "Evolutionary_biology", "Neurobiology", "Immunology", "Microbiology", "Biophysics",
                "Systems_biology", "Computational_biology", "Structural_biology", "Enzymology",
                
                # Mathematical Biology
                "Mathematical_and_theoretical_biology", "Bioinformatics", "Phylogenetics",
                
                # Advanced Physics
                "Theoretical_physics", "Particle_physics", "Relativity", "Cosmology", "String_theory",
                "Statistical_mechanics", "Condensed_matter_physics"
            ],
            "search_batches": search_batches,
            "save_dir": "./wikipedia_specialized_dataset",
            "max_articles": 3000,
            "max_workers": 3,
            "request_delay": (1, 2),
            "retry_attempts": 3,
            "retry_delay": 3,
            "timeout": 30,
            "min_content_length": 1000,
            "max_content_length": 500000,
            "log_level": "INFO",
            "languages": ["en"]
        }

        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        else:
            try:
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2)
            except Exception as e:
                print(f"Could not save default config: {e}")
        
        return default_config

    def setup_logging(self):

        try:
            log_dir = os.path.join(self.config['save_dir'], 'logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f"wiki_specialized_crawler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
            
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            
            logging.basicConfig(
                level=getattr(logging, self.config['log_level']),
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file, encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            print(f"Logging setup failed: {e}")
            self.logger = logging.getLogger(__name__)

    def setup_directories(self):
   
        directories = [
            self.config['save_dir'],
            os.path.join(self.config['save_dir'], 'logs')
        ]
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                print(f"Could not create directory {directory}: {e}")

    def _create_session(self) -> requests.Session:

        session = requests.Session()
        session.headers.update({
            'User-Agent': 'SpecializedDatasetCrawler/1.0 (Educational/Research Purpose)',
            'Accept': 'application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        return session

    def _signal_handler(self, signum, frame):

        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self.running = False
        self._save_progress()
        sys.exit(0)

    def _save_progress(self):

        progress_data = {
            'visited_urls': list(self.visited_urls),
            'scraped_hashes': list(self.scraped_hashes),
            'failed_urls': self.failed_urls,
            'article_count': self.article_count
        }
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(progress_data, f)
            print(f"Progress saved. Articles scraped: {self.article_count}")
        except Exception as e:
            print(f"Failed to save progress: {e}")

    def _load_progress(self):
  
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'rb') as f:
                    progress_data = pickle.load(f)
                self.visited_urls = set(progress_data.get('visited_urls', []))
                self.scraped_hashes = set(progress_data.get('scraped_hashes', []))
                self.failed_urls = progress_data.get('failed_urls', {})
                self.article_count = progress_data.get('article_count', 0)
                print(f"Resumed from previous session. Articles already scraped: {self.article_count}")
            except Exception as e:
                print(f"Failed to load progress: {e}")

    def search_wikipedia_articles(self, search_term: str, limit: int = 50) -> List[str]:

        try:
            params = {
                'action': 'opensearch',
                'search': search_term,
                'limit': limit,
                'namespace': 0,
                'format': 'json'
            }
            
            delay = random.uniform(*self.config['request_delay'])
            time.sleep(delay)
            
            response = self.session.get(self.wiki_search_url, params=params, timeout=self.config['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                if len(data) >= 4:
                    urls = data[3]
                    return urls
            
        except Exception as e:
            print(f"Error searching for '{search_term}': {e}")
        
        return []

    def get_category_articles(self, category: str, limit: int = 100) -> List[str]:
 "
        try:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Category:{category}',
                'cmlimit': limit,
                'cmnamespace': 0,
                'format': 'json'
            }
            
            delay = random.uniform(*self.config['request_delay'])
            time.sleep(delay)
            
            response = self.session.get(self.wiki_search_url, params=params, timeout=self.config['timeout'])
            
            if response.status_code == 200:
                data = response.json()
                if 'query' in data and 'categorymembers' in data['query']:
                    articles = []
                    for member in data['query']['categorymembers']:
                        title = member['title'].replace(' ', '_')
                        url = f"https://en.wikipedia.org/wiki/{quote(title)}"
                        articles.append(url)
                    return articles
            
        except Exception as e:
            print(f"Error getting category '{category}': {e}")
        
        return []

    def clean_text_for_llm(self, text: str) -> str:
  
        if not text:
            return ""
        
        # Remove citation patterns like [1], [2], [citation needed], etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text)
        text = re.sub(r'\[clarification needed\]', '', text)
        text = re.sub(r'\[when\?\]', '', text)
        text = re.sub(r'\[who\?\]', '', text)
        text = re.sub(r'\[where\?\]', '', text)
        text = re.sub(r'\[why\?\]', '', text)
        text = re.sub(r'\[how\?\]', '', text)
        text = re.sub(r'\[dubious.*?\]', '', text)
        text = re.sub(r'\[verify.*?\]', '', text)
        text = re.sub(r'\[original research\?\]', '', text)
        text = re.sub(r'\[POV\]', '', text)
        text = re.sub(r'\[according to whom\?\]', '', text)
        
    
        text = re.sub(r'\{\{.*?\}\}', '', text, flags=re.DOTALL)  # Remove templates
        text = re.sub(r'\[\[File:.*?\]\]', '', text)  # Remove file links
        text = re.sub(r'\[\[Image:.*?\]\]', '', text)  # Remove image links
        text = re.sub(r'\[\[Category:.*?\]\]', '', text)  # Remove category links
        
        # Clean up internal links 
        text = re.sub(r'\[\[([^|\]]+)\|([^|\]]+)\]\]', r'\2', text) 
        text = re.sub(r'\[\[([^|\]]+)\]\]', r'\1', text)  
        
        # Remove external links 
        text = re.sub(r'\[http[s]?://[^\s\]]+\s+([^\]]+)\]', r'\1', text)
        text = re.sub(r'\[http[s]?://[^\s\]]+\]', '', text)
        
        # Clean up HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = text.strip()
        
        return text
    
    def categorize_domain(self, title: str, categories: List[str]) -> str:
 
        title_lower = title.lower()
        categories_text = ' '.join(categories).lower()
    
        def contains(keywords):
            return any(kw in title_lower or kw in categories_text for kw in keywords)
    
        # Quantum Physics
        if contains(['quantum', 'qubit', 'superposition', 'entanglement', 'decoherence', 'wave function',
                     'schrödinger', 'heisenberg', 'bell', 'epr', 'copenhagen', 'many-worlds']):
            return "Quantum Physics"
    
        # Mathematics & Theorems
        if contains(['theorem', 'conjecture', 'proof', 'mathematics', 'algebra', 'topology', 'geometry',
                     'number theory', 'analysis', 'calculus', 'differential', 'integration', 'prime',
                     'fermat', 'gödel', 'riemann', 'poincaré', 'graph theory', 'combinatorics']):
            return "Mathematics & Theorems"
    
        # Molecular Biology
        if contains(['dna', 'rna', 'protein', 'enzyme', 'gene', 'chromosome', 'transcription', 'translation',
                     'replication', 'molecular biology', 'biochemistry', 'cell biology', 'genetics']):
            return "Molecular Biology"
    
        # Neurobiology & Biophysics
        if contains(['neuron', 'brain', 'neural', 'synapse', 'neurotransmitter', 'action potential',
                     'biophysics', 'membrane', 'ion channel', 'receptor', 'signal transduction']):
            return "Neurobiology & Biophysics"
    
        # Evolutionary & Developmental Biology
        if contains(['evolution', 'natural selection', 'phylogeny', 'development', 'embryo', 'stem cell',
                     'differentiation', 'morphogenesis', 'adaptation', 'speciation']):
            return "Evolutionary & Developmental Biology"
    
        # Immunology & Microbiology
        if contains(['immune', 'antibody', 'antigen', 'lymphocyte', 'bacteria', 'virus', 'pathogen',
                     'infection', 'vaccination', 'microbiology', 'immunology']):
            return "Immunology & Microbiology"
    
        # Mathematical Biology & Bioinformatics
        if contains(['bioinformatics', 'computational biology', 'systems biology', 'mathematical biology',
                     'population dynamics', 'epidemiology', 'phylogenetics', 'genomics']):
            return "Mathematical Biology & Bioinformatics"
    
        # Theoretical Physics
        if contains(['relativity', 'cosmology', 'string theory', 'particle physics', 'field theory',
                     'gauge theory', 'supersymmetry', 'black hole', 'big bang', 'higgs']):
            return "Theoretical Physics"
    
        return "Specialized Science"

    def extract_article_content(self, url: str) -> Optional[GeneralArticleData]:

        if not self.running or url in self.visited_urls:
            return None
            
        try:
            delay = random.uniform(*self.config['request_delay'])
            time.sleep(delay)
            
            response = self.session.get(url, timeout=self.config['timeout'])
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = self._extract_title(soup)
            if not title:
                return None
            
            # Skip disambiguation pages and other meta pages
            if any(skip_term in title.lower() for skip_term in ['disambiguation', 'list of', 'category:', 'template:']):
                return None
            
            # Extract main content
            content = self._extract_content(soup)
            if not content or len(content) < self.config['min_content_length']:
                return None
            
            # Clean content for LLM training
            cleaned_content = self.clean_text_for_llm(content)
            if not cleaned_content or len(cleaned_content) < self.config['min_content_length']:
                return None
            
            # Check for duplicate content
            content_hash = hashlib.md5(cleaned_content.encode('utf-8')).hexdigest()
            if content_hash in self.scraped_hashes:
                return None
            
            # Extract metadata
            categories = self._extract_categories(soup)
            category = categories[0] if categories else "Specialized"
            domain = self.categorize_domain(title, categories)
            summary = self.clean_text_for_llm(self._extract_summary(soup))
            
            # Create article ID
            with self.lock:
                article_id = f"WIKI_SPECIALIZED_{self.article_count + 1:06d}"
            
            article_data = GeneralArticleData(
                article_id=article_id,
                title=title,
                url=url,
                content=cleaned_content,
                category=category,
                summary=summary,
                content_hash=content_hash,
                scraped_at=datetime.now().isoformat(),
                word_count=len(cleaned_content.split()),
                language="en",
                domain=domain
            )
            
            with self.lock:
                self.scraped_hashes.add(content_hash)
                self.visited_urls.add(url)
            
            return article_data
            
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return None

    def _extract_title(self, soup: BeautifulSoup) -> str:

        title_elem = soup.find('h1', {'class': 'firstHeading'})
        if title_elem:
            return title_elem.get_text().strip()
        
        title_elem = soup.find('title')
        if title_elem:
            title = title_elem.get_text().strip()
            return title.replace(' - Wikipedia', '')
        
        return ""

    def _extract_content(self, soup: BeautifulSoup) -> str:

        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if not content_div:
            return ""
        
        # Remove unwanted elements
        for unwanted in content_div.find_all([
            'table', 'div', 'span'
        ], {
            'class': [
                'navbox', 'infobox', 'ambox', 'metadata', 'hatnote', 'dablink',
                'mbox', 'plainlinks', 'sister-projects', 'reflist', 'references',
                'cite', 'citation', 'refbegin', 'refend'
            ]
        }):
            unwanted.decompose()
        
        # Remove specific elements
        for element in content_div.find_all(['sup', 'sub']):
            if element.get_text().strip():
                element.replace_with(element.get_text())
            else:
                element.decompose()
        
        # Remove navigation elements and reference sections
        for nav in content_div.find_all(['div'], {'id': ['toc', 'References', 'External_links', 'See_also', 'Notes']}):
            nav.decompose()
        
        # Extract text from paragraphs and headers
        content_parts = []
        
        for element in content_div.find_all(['p', 'h2', 'h3', 'h4', 'ul', 'ol']):
            if element.name in ['h2', 'h3', 'h4']:
                header_text = element.get_text().strip()
                if header_text and len(header_text) > 2:
                    content_parts.append(f"\n{header_text}\n")
            elif element.name == 'p':
                text = element.get_text().strip()
                if len(text) > 30:
                    content_parts.append(text)
            elif element.name in ['ul', 'ol']:
                # Include list items for comprehensive knowledge
                list_items = []
                for li in element.find_all('li'):
                    item_text = li.get_text().strip()
                    if len(item_text) > 10:
                        list_items.append(f"• {item_text}")
                if list_items:
                    content_parts.append('\n'.join(list_items))
        
        return '\n\n'.join(content_parts)

    def _extract_categories(self, soup: BeautifulSoup) -> List[str]:

        categories = []
        category_links = soup.find_all('a', href=re.compile(r'/wiki/Category:'))
        
        for link in category_links[:5]:  # Limit to first 5 categories
            category = link.get_text().strip()
            if category and len(category) > 2:
                categories.append(category)
        
        return categories

    def _extract_summary(self, soup: BeautifulSoup) -> str:
   
        content_div = soup.find('div', {'class': 'mw-parser-output'})
        if content_div:
            first_p = content_div.find('p')
            if first_p:
                summary = first_p.get_text().strip()
                return summary[:1000] + "..." if len(summary) > 1000 else summary
        return ""

    def append_to_training_file(self, article: GeneralArticleData):
  
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                # Write article header
                f.write("=" * 100 + "\n")
                f.write(f"TITLE: {article.title}\n")
                f.write(f"DOMAIN: {article.domain}\n")
                f.write(f"CATEGORY: {article.category}\n")
                f.write(f"ARTICLE_ID: {article.article_id}\n")
                f.write(f"WORD_COUNT: {article.word_count}\n")
                f.write(f"SOURCE: {article.url}\n")
                f.write("=" * 100 + "\n\n")
                
                # Write summary if available
                if article.summary:
                    f.write("SUMMARY:\n")
                    f.write(article.summary)
                    f.write("\n\n")
                
                # Write main content
                f.write("CONTENT:\n")
                f.write(article.content)
                f.write("\n\n")
                
                # Add separator between articles
                f.write("\n" + ">" * 50 + " END OF ARTICLE " + "<" * 50 + "\n\n\n")
                
        except Exception as e:
            print(f"Failed to append article {article.article_id} to training file: {e}")

    def crawl_wikipedia_specialized_content(self):

        print(f"Starting Wikipedia specialized content crawl. Target: {self.config['max_articles']} articles")
        print(f"Focus: Quantum Physics, Mathematics Theorems, Biology")
        print(f"Output file: {self.output_file}")
        
        # Initialize the output file with header
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write("WIKIPEDIA SPECIALIZED KNOWLEDGE TRAINING DATASET\n")
            f.write("Focus: Quantum Physics, Mathematics Theorems, Biological Concepts\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Target Articles: {self.config['max_articles']}\n")
            f.write("=" * 100 + "\n\n")
        
        all_article_urls = set()
        
        # 1. Search for articles using specialized search terms
        print("Searching for articles using specialized search terms...")
        for batch in self.config['search_batches']:
            for search_term in batch:
                if not self.running:
                    break
                print(f"Searching for: {search_term}")
                urls = self.search_wikipedia_articles(search_term, limit=30)
                all_article_urls.update(urls)
                time.sleep(1)
        
        # 2. Get articles from specialized categories
        print("Getting articles from specialized categories...")
        for category in self.config['specialized_categories']:
            if not self.running:
                break
            print(f"Getting articles from category: {category}")
            urls = self.get_category_articles(category, limit=50)
            all_article_urls.update(urls)
            time.sleep(1)
        
        print(f"Found {len(all_article_urls)} unique article URLs")
        
        # 3. Process articles
        article_urls = list(all_article_urls)
        random.shuffle(article_urls)
        
        progress_bar = tqdm(total=min(len(article_urls), self.config['max_articles']), 
                           desc="Processing articles")
        
        domain_counts = {}
        
        for url in article_urls:
            if (self.article_count >= self.config['max_articles'] or 
                not self.running):
                break
                
            if url not in self.visited_urls:
                article = self.extract_article_content(url)

                if article:
                    self.append_to_training_file(article)
                    self.article_count += 1
                    
                    # Track domain distribution
                    domain_counts[article.domain] = domain_counts.get(article.domain, 0) + 1
                    
                    progress_bar.set_description(f"Saved: {article.title[:40]}...")
                    progress_bar.update(1)
                    
                    # Save progress periodically
                    if self.article_count % 20 == 0:
                        self._save_progress()
                        print(f"\nDomain distribution so far: {domain_counts}")
        
        progress_bar.close()
        print(f"\nFinal domain distribution: {domain_counts}")

    def create_final_summary(self):

        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write("\n" + "=" * 100 + "\n")
                f.write("DATASET SUMMARY\n")
                f.write("=" * 100 + "\n")
                f.write(f"Total Articles Scraped: {self.article_count}\n")
                f.write(f"Completion Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Source: Wikipedia Specialized Knowledge Content\n")
                f.write(f"Content Domains: Quantum Physics, Mathematics & Theorems, Molecular Biology, Theoretical Physics\n")
                f.write(f"Language: English\n")
                f.write(f"Purpose: Specialized LLM Training Data\n")
                f.write("=" * 100 + "\n")
        except Exception as e:
            print(f"Error creating final summary: {e}")

    def create_domain_statistics(self):

        try:
            stats_file = os.path.join(self.config['save_dir'], 'dataset_statistics.json')
            
            # Count domain distribution
            domain_counts = {}
            total_words = 0
            
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract domain information from the file
                domain_matches = re.findall(r'DOMAIN: (.+)', content)
                word_count_matches = re.findall(r'WORD_COUNT: (\d+)', content)
                
                for domain in domain_matches:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                
                for word_count in word_count_matches:
                    total_words += int(word_count)
            
            stats = {
                'total_articles': self.article_count,
                'total_words': total_words,
                'average_words_per_article': total_words / self.article_count if self.article_count > 0 else 0,
                'domain_distribution': domain_counts,
                'collection_date': datetime.now().isoformat(),
                'data_source': 'Wikipedia',
                'language': 'English',
                'specialization': 'Quantum Physics, Mathematics Theorems, Biological Concepts'
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            print(f"Dataset statistics saved to: {stats_file}")
            return stats
            
        except Exception as e:
            print(f"Error creating statistics: {e}")
            return {}

    def run(self):
    
        print("Starting Wikipedia Specialized Knowledge Crawler for LLM Training")
        print("Specialization: Quantum Physics, Mathematics Theorems, Biological Concepts")
        print(f"Target: {self.config['max_articles']} articles")
        print(f"Already scraped: {self.article_count} articles")
        
        try:
            self.crawl_wikipedia_specialized_content()
        except KeyboardInterrupt:
            print("\nCrawling interrupted by user")
        except Exception as e:
            print(f"Unexpected error during crawling: {e}")
        finally:
            self._save_progress()
            self.create_final_summary()
            stats = self.create_domain_statistics()
            
            print(f"\nCrawling completed. Total articles scraped: {self.article_count}")
            print(f"Training data saved to: {self.output_file}")
            
            if stats:
                print(f"Total words collected: {stats.get('total_words', 0):,}")
                print(f"Average words per article: {stats.get('average_words_per_article', 0):.1f}")
                print("Domain distribution:")
                for domain, count in stats.get('domain_distribution', {}).items():
                    percentage = (count / self.article_count * 100) if self.article_count > 0 else 0
                    print(f"  {domain}: {count} articles ({percentage:.1f}%)")


def main():

    try:
        # You can customize the config file path if needed
        crawler = WikipediaSpecializedCrawler("wiki_specialized_crawler_config.json")
        crawler.run()
    except Exception as e:
        print(f"Failed to start crawler: {e}")

if __name__ == "__main__":
    main()
