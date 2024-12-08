import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import logging
import json
import pylatex
import subprocess

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
        
    def filter_data(self, df, exclude_preprints=False):
        if exclude_preprints:
            total_len = len(df)
            df['citation'] = df['citation'].fillna('').astype(str)
            df = df[~df['citation'].str.contains('rxiv', case=False, na=False)]
            logger.info(f"Filtered out {total_len - len(df)} papers from Rxiv.")
        return df

    def analyze_csv(self, csv_path, exclude_preprints=False):
        df = pd.read_csv(csv_path)
        df = self.filter_data(df, exclude_preprints)
        summary_json = self.summarize_data(df, exclude_preprints=exclude_preprints)
        self.generate_plots_with_percentiles(summary_json)
        return summary_json

    def summarize_data(self, df, exclude_preprints=False):
        output_json = "author_summary_no_preprint.json" if exclude_preprints else "author_summary.json"
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
            'preprints_excluded': exclude_preprints,
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
            'summary_json': output_json,
            'n_papers_plot': "author_n_papers_no_preprint.png" if exclude_preprints else "author_n_papers.png",
            'n_citations_plot': "author_citations_no_preprint.png" if exclude_preprints else "author_citations.png",   
            'label': "No Preprints" if exclude_preprints else "Full"
        }

        with open(output_json, 'w') as file:
            json.dump(summary_json, file, indent=4)

        logger.info(f"Summary saved to {output_json}")
        return summary_json

    def generate_plots_with_percentiles(self, summary_json):

        self.plot_bar_chart(
            summary_json['papers_per_author'],
            "Number of papers by author in 2024",
            "Number of papers",
            summary_json['paper_percentiles'],
            summary_json['n_papers_plot'],
            summary_json['specific_author']
        )
        self.plot_bar_chart(
            summary_json['citations_per_author'],
            "Total citations by author in 2024",
            "Total citations",
            summary_json['citation_percentiles'],
            summary_json['n_citations_plot'],
            summary_json['specific_author']
        )

    def plot_bar_chart(self, data_dict, title, ylabel, percentiles_dict, output_file, highlight_author=None, exclude_preprints=False):
        if exclude_preprints:
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
    
    @staticmethod
    def generate_latex_report(summary_list):
        def add_summary_section(doc, title, summary):
            doc.append(pylatex.Section(title))
            doc.append(pylatex.NoEscape(
                r'\textbf{Number of authors:} ' + str(summary['N_authors']) + r'\\'
            ))
            doc.append(pylatex.NoEscape(
                r'\textbf{Preprints excluded:} ' + str(summary.get('preprints_excluded', 'N/A')) + r'\\'
            ))
            doc.append(pylatex.NoEscape(
                r'\textbf{Specific author:} ' + str(summary['specific_author']) + r'\\'
            ))
            doc.append(pylatex.NoEscape(
                r'\textbf{Specific author papers:} ' + str(summary['specific_author_n_papers']) + r'\\'
            ))
            doc.append(pylatex.NoEscape(
                r'\textbf{Specific author citations:} ' + str(summary['specific_author_n_citations']) + r'\\'
            ))
            doc.append(pylatex.NoEscape(
                r'\textbf{Specific author papers percentile:} ' + str(summary['specific_n_papers_percentile']) + r'\\'
            ))
            doc.append(pylatex.NoEscape(
                r'\textbf{Specific author citations percentile:} ' + str(summary['specific_n_citations_percentile']) + r'\\'
            ))

        def add_plot(doc, image_path, caption):
            with doc.create(pylatex.Figure(position='h!')) as plot:
                plot.add_image(image_path, width=pylatex.utils.NoEscape(r'0.8\textwidth'))
                plot.add_caption(caption)

        # Main LaTeX Document
        doc = pylatex.Document()
        doc.preamble.append(pylatex.Command('title', 'Buck Author Publications Analysis Report'))
        doc.preamble.append(pylatex.Command('author', 'MBelic'))
        doc.preamble.append(pylatex.Command('date', pylatex.NoEscape(r'\today')))
        doc.append(pylatex.NoEscape(r'\maketitle'))

        # Add summaries for all JSON objects
        for idx, summary in enumerate(summary_list, start=1):
            title = f"Summary {idx} ({summary.get('label', 'No Label')})"
            add_summary_section(doc, title, summary)

        doc.append(pylatex.Section('Plots'))
        for idx, summary in enumerate(summary_list, start=1):
            plot_label = summary.get('label', f'Summary {idx}')
            add_plot(doc, summary['n_papers_plot'], f'Number of papers by author in 2024 ({plot_label})')
            add_plot(doc, summary['n_citations_plot'], f'Total citations by author in 2024 ({plot_label})')

        # Generate PDF
        doc.generate_pdf('author_report', clean_tex=False)
        
        # Compile PDF
        subprocess.run(['pdflatex', 'author_report.tex'])
        logger.info("Latex report generated.")
        return "author_report.tex"
    
    @staticmethod
    def compile_latex(texfile):
        subprocess.run(['pdflatex', texfile])
        logger.info("Latex report compiled.")



if __name__ == "__main__":
    file_to_analyze = "author_publications_20241207205456_Plus_Manual.csv"
    config_path = "author_config.cfg"
    analyzer = AuthorAnalyzer(config_path=config_path)
    
    summary_full = analyzer.analyze_csv(
        csv_path=file_to_analyze,
        exclude_preprints=False
        )

    summary_no_preprint = analyzer.analyze_csv(
        csv_path=file_to_analyze,
        exclude_preprints=True
        )
    
    analyzer.generate_latex_report([summary_full, summary_no_preprint])
    

