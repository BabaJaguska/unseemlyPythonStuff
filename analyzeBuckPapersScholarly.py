from scholarly import scholarly
import logging
import time
import random
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

config = json.load(open("author_config.cfg"))
authors_fullname = config['authors_fullname']
authors_no_scholar = config['authors_no_scholar']
authors_no_affiliation = config['authors_no_affiliation']


def fetch_author_papers(author_name, affiliation_filter="Buck Institute"):
    try:
        search_query = scholarly.search_author(author_name)
        for author in search_query:  # Loop through all matching authors
            if author_name in authors_no_affiliation:
                logger.info(f"Author '{author_name}' no affiliation, returning first author found.")
                scholarly.fill(author)
                return author['publications']
            if affiliation_filter.lower() in author.get('affiliation', '').lower():
                logger.info(f"Author '{author_name}' verified with affiliation: {author['affiliation']}")
                scholarly.fill(author)  # Retrieve full profile and publications
                return author['publications']
        logger.warning(f"No affiliation with '{affiliation_filter}' found for '{author_name}' or no profile found.")
        return []
    except StopIteration:
        logger.warning(f"No data found for author: {author_name}")
        return []

def filter_papers_by_year(papers, year_threshold):
    return [paper for paper in papers if 'pub_year' in paper['bib'] and int(paper['bib']['pub_year']) >= year_threshold]

def scrape_publications():
    year_threshold = 2024
    affiliation_filter = "Buck Institute"
    all_filtered_papers = []

    for author in authors_fullname:
        print(f"Fetching publications for {author}")
        papers = fetch_author_papers(author, affiliation_filter)
        filtered_papers = filter_papers_by_year(papers, year_threshold)
        filtered_papers_bib_plus = [
            {   "author": author,
                **paper['bib'],  # Unpack all keys from the bib dictionary
                "num_citations": paper.get('num_citations', 0)  # Add num_citations as an additional key
            }
            for paper in filtered_papers
        ]
    
        all_filtered_papers.extend(filtered_papers_bib_plus)
        logger.info(f"Found {len(filtered_papers_bib_plus)} papers for {author} in {year_threshold} or later.")
        rand_sleep_time = random.randint(4, 8)
        logger.debug(f"--------------Sleeping for {rand_sleep_time} seconds...")
        time.sleep(rand_sleep_time)


    logger.debug(f"\nTotal papers from {year_threshold} or later: {len(all_filtered_papers)}\n")
    df = pd.DataFrame(all_filtered_papers)
    logger.debug("df: ", df.head())
    df.to_csv("buck_papers.csv", index=False)

def analyze_csv(csv_path):
    N_authors_intended = len(authors_fullname)
    df = pd.read_csv(csv_path)
    N_total = len(df)
    logger.info(f"Total papers: {N_total}")
    unique_authors = df['author'].unique()
    N_authors = len(unique_authors)
    logger.info(f"Total authors: {N_authors}")
    # Count papers by author
    author_counts = df['author'].value_counts()
    logger.info(f"Author counts: {author_counts}")
    df_auth = df.groupby('author').agg({'num_citations': 'sum'}).sort_values('num_citations', ascending=False)
    logger.info(f"Author citations: {df_auth}")

    percentiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    citations_percentiles = df_auth['num_citations'].quantile(percentiles)
    logger.info(f"Citations percentiles:\n{citations_percentiles}")

    num_papers_percentiles = author_counts.quantile(percentiles)
    logger.info(f"Number of papers percentiles:\n{num_papers_percentiles}")

    which_percentile_num_papers_is_david = scipy.stats.percentileofscore(author_counts, author_counts["David Furman"])
    which_percentile_num_papers_is_david = round(which_percentile_num_papers_is_david)
    logger.info(f"David Furman is at the {which_percentile_num_papers_is_david} percentile in terms of number of papers.")
    which_percentile_citations_is_david = scipy.stats.percentileofscore(df_auth['num_citations'], df_auth.loc["David Furman", "num_citations"])
    which_percentile_citations_is_david = round(which_percentile_citations_is_david)
    logger.info(f"David Furman is at the {which_percentile_citations_is_david} percentile in terms of number of citations.")

    # Plot author counts
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    authors = author_counts.index
    colors = ['lightblue' if author == "David Furman" else 'blue' for author in authors]
    plt.bar(author_counts.index, author_counts.values, color=colors)
    plt.xticks(rotation=60)
    plt.xlabel("Authors")
    plt.ylabel("Number of papers in 2024")
    plt.title("Number of papers by author in 2024")
    plt.tight_layout()
    plt.grid(axis='y')
    for i, count in enumerate(author_counts.values):
        if author_counts.index[i] == "David Furman":
            plt.text(i, count, f"{count}\n($P_{{{which_percentile_num_papers_is_david}}}$)", ha='center', va='bottom')
        else:
            plt.text(i, count, str(count), ha='center', va='bottom')
    # plot percentile lines
    plt.axline((0, num_papers_percentiles[0.50]), (N_authors, num_papers_percentiles[0.50]),
                color='red', linestyle='--', label='median')
    plt.axline((0, num_papers_percentiles[0.75]), (N_authors, num_papers_percentiles[0.75]),
                color='green', linestyle='--', label='75th percentile')
    plt.axline((0, num_papers_percentiles[0.95]), (N_authors, num_papers_percentiles[0.95]),
                color='purple', linestyle='--', label='95th percentile')
    plt.legend()
    # write vallues on percentile lines
    offset_num = config['n_papers_plot_offset']
    plt.text(N_authors, num_papers_percentiles[0.50]-offset_num,
                f"{num_papers_percentiles[0.50]:.0f}", ha='left', va='center', color='red')
    plt.text(N_authors, num_papers_percentiles[0.75]-offset_num,
                f"{num_papers_percentiles[0.75]:.0f}", ha='left', va='center', color='green')
    plt.text(N_authors, num_papers_percentiles[0.95]-offset_num,
                f"{num_papers_percentiles[0.95]:.0f}", ha='left', va='center', color='purple')
    
    

    plt.subplot(2, 1, 2)
    colors = ['lightblue' if author == "David Furman" else 'blue' for author in df_auth.index]
    plt.bar(df_auth.index, df_auth['num_citations'], color=colors)
    plt.xticks(rotation=60)
    plt.xlabel("Authors")
    plt.ylabel("Total citations in 2024")
    plt.title("Total citations by author in 2024")
    plt.tight_layout()
    plt.grid(axis='y')
    for i, count in enumerate(df_auth['num_citations']):
        if df_auth.index[i] == "David Furman":
            plt.text(i, count, f"{count}\n($P_{{{which_percentile_citations_is_david}}}$)", ha='center', va='bottom')
        else:
            plt.text(i, count, str(count), ha='center', va='bottom')
    # plot percentile lines
    plt.axline((0, citations_percentiles[0.50]), (N_authors, citations_percentiles[0.50]),
                color='red', linestyle='--', label='median')
    plt.axline((0, citations_percentiles[0.75]), (N_authors, citations_percentiles[0.75]),
                color='green', linestyle='--', label='75th percentile')
    plt.axline((0, citations_percentiles[0.95]), (N_authors, citations_percentiles[0.95]),
                color='purple', linestyle='--', label='95th percentile')
    plt.legend()
    # write vallues on percentile lines
    offset = config['n_citations_plot_offset']
    plt.text(N_authors, citations_percentiles[0.50]-offset,
              f"{citations_percentiles[0.50]:.0f}", ha='left', va='center', color='red')
    plt.text(N_authors, citations_percentiles[0.75]-offset,
              f"{citations_percentiles[0.75]:.0f}", ha='left', va='center', color='green')
    plt.text(N_authors, citations_percentiles[0.95]-offset,
              f"{citations_percentiles[0.95]:.0f}", ha='left', va='center', color='purple')

    plt.savefig("author_counts.png")


    n_authors_missing = N_authors_intended - N_authors
    logger.info(f"Authors missing: {n_authors_missing}")
    N_authors_no_profile = len(authors_no_scholar)
    logger.info(f"Authors with no Google Scholar profile: {N_authors_no_profile}")
    missing_authors = [author for author in authors_fullname if author not in unique_authors]
    logger.info(f"Missing authors: {missing_authors}")
    authors_with_profiles_but_no_papers = [author for author in authors_fullname if author not in unique_authors and author not in authors_no_scholar]
    logger.info(f"Authors with profiles but no papers: {authors_with_profiles_but_no_papers}")
if __name__ == "__main__":
    csv_path = "buck_papers.csv"
    analyze_csv(csv_path)
