import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # Dictionary to store probability of each page
    probability_distribution = dict()

    # Find all pages linked to this page
    linked_pages = corpus[page]

    # Probability for choosing all pages
    for page in corpus:
        probability_distribution[page] = (1 - damping_factor) / len(corpus)

    # Probability for choosing one of the linked page
    for page in linked_pages:
        probability_distribution[page] += damping_factor / len(linked_pages)
    
    return probability_distribution

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Dictionary to store the rank of each page
    pagerank = dict()

    # Initialize all the ranks to 0
    for page in corpus:
        pagerank[page] = 0
    
    # Choose a random page for the first sample
    sample = random.choice(list(corpus))
    pagerank[sample] += 1 / n

    for i in range(1, n):
        probability_distribution = transition_model(corpus, sample, damping_factor)

        # Keep track of pages and their probabilities
        next_pages = list(probability_distribution.keys())
        probabilties = list(probability_distribution.values())

        # Pick a random page based on their probabilities
        sample = random.choices(next_pages, weights=probabilties)[0]

        # Update the sample page's rank
        pagerank[sample] += 1 / n 
    
    return pagerank

    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Dictionary to store the rank of each page
    pagerank = dict()

    # Store the total number of pages
    N = len(corpus)

    # Assign each page a rank of 1/N
    for page in corpus:
        pagerank[page] = 1 / N
    
    while True:

        # Keep count of the pages whose rank does not change by 0.001
        flags = 0

        # Loop over each page
        for page in corpus:

            new_rank = 0

            for linked_page in corpus:

                # Find all pages which have a link to current page
                if page in corpus[linked_page]:

                    # Calculate the rank using the formula
                    new_rank += pagerank[linked_page] / len(corpus[linked_page])

                # If a page has no links, assume it has a link to each page
                elif len(corpus[linked_page]) == 0:
                    new_rank += pagerank[linked_page] / N

            # Complete the remaining calculation
            new_rank *= damping_factor
            new_rank += (1 - damping_factor) / N

            # Check if the ranks change by a factor of 0.001
            if abs(pagerank[page] - new_rank) < 0.001:
                flags += 1
            
            pagerank[page] = new_rank

        # Exit if the ranks do not change by 0.001 for each page
        if flags == N:
            break
    
    return pagerank

    raise NotImplementedError


if __name__ == "__main__":
    main()
