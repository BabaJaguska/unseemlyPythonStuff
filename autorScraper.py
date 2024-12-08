import time
import random
import json
import pandas as pd
import logging
from scholarly import scholarly
from time import localtime, strftime

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AuthorScraper:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.authors_fullname = self.config['authors_fullname']
        self.authors_no_affiliation = self.config['authors_no_affiliation']
        self.since_year = self.config['since_year']

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)

    def fetch_author_papers(self, author_name, affiliation_filter="Buck Institute"):
        try:
            search_query = scholarly.search_author(author_name)
            for author in search_query:
                if author_name in self.authors_no_affiliation:
                    logger.info(f"Author '{author_name}' has no affiliation, using first match.")
                    scholarly.fill(author)
                    return author.get('publications', [])
                if affiliation_filter.lower() in author.get('affiliation', '').lower():
                    logger.info(f"Author '{author_name}' verified with affiliation: {author['affiliation']}")
                    scholarly.fill(author)
                    return author.get('publications', [])
            logger.warning(f"No matching affiliation for '{author_name}' or profile not found.")
            return []
        except StopIteration:
            logger.warning(f"No data found for author: {author_name}")
            return []

    @staticmethod
    def filter_papers_by_year(papers, year_threshold):
        return [paper for paper in papers if 'pub_year' in paper['bib'] and int(paper['bib']['pub_year']) >= year_threshold]

    def scrape_publications(self, affiliation_filter="Buck Institute"):
        tm = strftime("%Y%m%d%H%M%S", localtime())
        output_csv = f"author_publications_{tm}.csv"
        logger.info(f"Scraping publications... Saving to {output_csv}")
        all_filtered_papers = []
        for author in self.authors_fullname:
            logger.info(f"Fetching publications for {author}")
            papers = self.fetch_author_papers(author, affiliation_filter)
            filtered_papers = self.filter_papers_by_year(papers, self.since_year)
            filtered_papers_bib_plus = [
                {
                    "author": author,
                    **paper['bib'],
                    "num_citations": paper.get('num_citations', 0)
                }
                for paper in filtered_papers
            ]
            all_filtered_papers.extend(filtered_papers_bib_plus)
            logger.info(f"Found {len(filtered_papers_bib_plus)} papers for {author} in {self.since_year} or later.")
            time.sleep(random.randint(4, 8))

        df = pd.DataFrame(all_filtered_papers)
        df.to_csv(output_csv, index=False)
        logger.info(f"Data saved to {output_csv}")

if __name__ == "__main__":
    scraper = AuthorScraper(config_path="author_config.cfg")
    scraper.scrape_publications()
