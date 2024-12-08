import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AuthorAnalyzer:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
        self.authors_fullname = self.config['authors_fullname']
        self.authors_no_scholar = self.config['authors_no_scholar']
        self.specific_author = self.config['specific_author']

    @staticmethod
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
        
    def filter_data(self, df):
        if self.config['filter_out_arxiv']:
            total_len = len(df)
            df['citation'] = df['citation'].fillna('').astype(str)
            df = df[~df['citation'].str.contains('rxiv', case=False, na=False)]
            logger.info(f"Filtered out {total_len - len(df)} papers from Rxiv.")
        return df

    def analyze_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        df = self.filter_data(df)
        summary_json = self.summarize_data(df)
        self.generate_plots_with_percentiles(summary_json)

    def summarize_data(self, df):
        output_json = "author_summary.json" if self.config['filter_out_arxiv'] else "author_summary_no_preprint.json"
        unique_authors = df['author'].unique()
        N_authors = len(unique_authors)
        missing_authors = [author for author in self.authors_fullname if author not in unique_authors]
        author_counts = df['author'].value_counts()
        for author in missing_authors:
            author_counts[author] = 0
        df_auth = df.groupby('author')['num_citations'].sum().sort_values(ascending=False)
        for author in missing_authors:
            df_auth[author] = 0

        percentiles = self.config.get('percentiles', [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        papers_percentiles = author_counts.quantile(percentiles).to_dict()
        citations_percentiles = df_auth.quantile(percentiles).to_dict()

        # Percentile of a highlighted author
        if self.specific_author in author_counts:
            paper_percentile = scipy.stats.percentileofscore(author_counts, author_counts[self.specific_author])
            citation_percentile = scipy.stats.percentileofscore(df_auth, df_auth[self.specific_author])
        else:
            paper_percentile = citation_percentile = None

        summary_json = {
            'N_authors': N_authors,
            'papers_per_author': author_counts.to_dict(),
            'citations_per_author': df_auth.to_dict(),
            'paper_percentiles': {str(k): float(round(v,2)) for k, v in papers_percentiles.items()},
            'citation_percentiles': {str(k): float(round(v,2)) for k, v in citations_percentiles.items()},
            'authors_with_papers': unique_authors.tolist(),
            'authors_not_found': missing_authors,
            'specific_author': self.specific_author,
            'specific_author_n_papers': int(author_counts.get(self.specific_author, 0)),
            'specific_author_n_citations': int(df_auth.get(self.specific_author, 0)),
            'specific_n_papers_percentile': round(float(paper_percentile), 2) if paper_percentile else None,
            'specific_n_citations_percentile': round(float(citation_percentile), 2) if citation_percentile else None,
        }

        with open(output_json, 'w') as file:
            json.dump(summary_json, file, indent=4)

        logger.info(f"Summary saved to {output_json}")
        return summary_json

    def generate_plots_with_percentiles(self, summary_json):

        self.plot_bar_chart(
            summary_json['papers_per_author'], "Number of papers by author in 2024", "Number of papers",
            summary_json['paper_percentiles'], "author_counts.png", summary_json['specific_author']
        )
        self.plot_bar_chart(
            summary_json['citations_per_author'], "Total citations by author in 2024", "Total citations",
            summary_json['citation_percentiles'], "author_citations.png", summary_json['specific_author']
        )

    def plot_bar_chart(self, data_dict, title, ylabel, percentiles_dict, output_file, highlight_author=None):
        if self.config.get('filter_out_arxiv', False):
            title += " (excluding Rxiv)"
            output_file = output_file.replace(".png", "_no_preprint.png")

        authors = list(data_dict.keys())
        values = list(data_dict.values())

        colors = ['lightblue' if author == highlight_author else 'blue' for author in authors]

        plt.figure(figsize=(12, 8))
        plt.bar(authors, values, color=colors)
        plt.xlabel("")
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=10)

        # Add percentile lines and labels
        offset = 0.03 * max(values)
        values_taken = set()
        for percentile_str, value in percentiles_dict.items():
            percentile_float = float(percentile_str)
            if int(value) in values_taken:
                offset *= -1
            plt.axhline(y=value, color='purple', linestyle='--', label=f'{percentile_float * 100:.0f}th Percentile')
            plt.text(len(authors) - 1, value - offset,
                    f"P$_{{{percentile_float * 100:.0f}}}$ = {value:.0f}",
                    ha='left', va='center', color='black', fontsize=11)
            values_taken.add(int(value))
            values_taken.add(int(value) + 1)
            values_taken.add(int(value) - 1)

        for i, value in enumerate(values):
            color = 'red' if authors[i] == highlight_author else 'black'
            plt.text(i, value, f"{value}",
                    ha='center', va='bottom',
                    fontsize=11, color=color)

        plt.tight_layout()
        plt.grid(axis='y')
        plt.savefig(output_file)
        plt.close()
        logger.info(f"Plot saved to {output_file}")
    

if __name__ == "__main__":
    analyzer = AuthorAnalyzer(config_path="author_config.cfg")
    analyzer.analyze_csv("author_publications_20241207205456_Plus_Manual.csv")
