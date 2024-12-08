import requests
import logging
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())


authors_fullname = [
    "Julie Andersen",
    "Chris Benz",
    "Pierre-Yves Desprez",
    "Lee Hood",
    "Francesca E. Duncan",
    "Lisa Ellerby",
    "David Furman",
    "Jennifer Garrison",
    "Pejmun Haghighi",
    "Malene Hansen",
    "Claudio Hetz",
    "Pankaj Kapahi",
    "Gordon Lithgow",
    "Simon Melov",
    "John Newman",
    "Nathan Price",
    "Birgit Schilling",
    "Tara Tracy",
    "Eric Verdin",
    "Ashley Webb",
    "Dan Winer",
    "Chuankai Zhou"
]

authors_abbrev = [
    "Andersen, J.",
    "Benz, C.",
    "Desprez, P-Y.",
    "Hood, L.",
    "Duncan, F.E.",
    "Ellerby, L.M.",
    "Furman, D.",
    "Garrison, J.L.",
    "Haghighi, P.",
    "Hansen, M.",
    "Hetz, C.",
    "Kapahi, P.",
    "Lithgow, G.J.",
    "Melov, S.",
    "Newman, J.C.",
    "Price, N.D.",
    "Schilling, B.",
    "Tracy, T.",
    "Verdin, E.",
    "Webb, A.E.",
    "Winer, D.",
    "Zhou, C."
]

authors_surnames = [
    "Andersen",
    "Benz",
    "Desprez",
    "Hood",
    "Duncan",
    "Ellerby",
    "Furman",
    "Garrison",
    "Haghighi",
    "Hansen",
    "Hetz",
    "Kapahi",
    "Lithgow",
    "Melov",
    "Newman",
    "Price",
    "Schilling",
    "Tracy",
    "Verdin",
    "Webb",
    "Winer",
    "Zhou"
]


def fetch_preprints_batch(start_date, end_date, cursor=0, server="bioRxiv"):
    '''Fatches preprints from bioRxiv or medRxiv for a given time interval
    Args:   start_date: str, start date in format "YYYY-MM-DD"
            end_date: str, end date in format "YYYY-MM-DD"
            cursor: int, cursor for batch start
            server: str, server name, either "bioRxiv" or "medRxiv"
    Returns: 
             data: list, list of preprints
    '''
    '''
        The following metadata elements are returned by the API:

        doi
        title
        authors
        author_corresponding
        author_corresponding_institution
        date
        version
        category
        jats xml path
        abstract
        published'''

    url = f"https://api.biorxiv.org/details/{server}/{start_date}/{end_date}/{cursor}"
    response = requests.get(url)
    logger.debug(f"Fetching data from {url}")
    logger.debug(f"Response code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        logger.error(f"Failed to fetch data from {server}, error code: {response.status_code}")
        return None
    
def filter_batch(data, relevant_authors):
    '''Filters preprints based on author
    Args:   data: list, list of preprints
            author: str, author name
    Returns: 
            filtered_data: list, list of preprints filtered by authors
    '''
    filtered_data = []
    for paper in data:
        authors = paper['authors']
        if any(relevant_author in authors for relevant_author in relevant_authors):
            filtered_data.append(paper)

    logger.debug(f"Total papers found in batch for relevant authors: {len(filtered_data)}")
    if filtered_data:
        logger.debug("Example: ", filtered_data)

    return filtered_data



def main():
    start_date = "2024-01-01"
    end_date = "2024-12-06"
    filtered_data = []
    cursor = 0
    server = "biorxiv"
    data = fetch_preprints_batch(start_date, end_date, cursor, server)
    total_num_papers = int(data['messages'][0]['total'])
    logger.info(f"Messages: {data['messages']}")

    for cursor in tqdm(range(0, total_num_papers, 100), desc="Fetching paper batch"):
        data = fetch_preprints_batch(start_date, end_date, cursor, server)
        if data:
            logger.info(f"Total papers found: {len(data['collection'])}")
            filtered_batch = filter_batch(data['collection'], authors_abbrev)
            filtered_data.extend(filtered_batch)

        else:
            logger.error("No data found")

    df = pd.DataFrame(filtered_data)
    df.to_csv("filtered_papers.csv", index=False)


if __name__ == "__main__":
    main()