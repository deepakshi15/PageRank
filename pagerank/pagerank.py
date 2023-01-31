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
    model_trans={}
    
    #no of files present in corpus
    no_of_files=len(corpus)
    
    #gives no of links from current page
    no_of_links=len(corpus[page])
    
    if no_of_links!=0:

        #calculates random probability which is applicable for all pages
        ran_probability=(1-damping_factor)/no_of_files

        #calculates specific page probability
        specific_probability=damping_factor/no_of_links
    
    else:

        #calculates random probability which is applicable for all pages
        ran_probability=(1-damping_factor)/no_of_files
 
        #calculates specific page probability
        specific_probability=0

    #iteration over files
    for file in corpus:

        #check whether if cuurent page has any links or not
        if len(corpus[page])==0:
            model_trans[file]=1/no_of_files
        else:

            #if file is not current page then there is no need to get its link
            if file not in corpus[page]:
                model_trans[file]=ran_probability

            else:
                #probability for linked pages is specific probability + random probability
                model_trans[file]=specific_probability+ran_probability

    #check if sum of both probabilities is 1
    if round(sum(model_trans.values()),5)!=1: 
        print(f'ERROR! probability adds to {sum(model_trans())}')
    return model_trans

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    sample_spaces={}

    for page in corpus:
        sample_spaces[page]=0
    
    #sample is initially none
    sam=None

    for i in range(n):

        if sam==None:
            probability=list(corpus.keys())
            sam=random.choice(probability)
            sample_spaces[sam]+=1
        
        else:

            another_sam_prob=transition_model(corpus,sam,damping_factor)

            probability=list(another_sam_prob.keys())

            wght=[another_sam_prob[key] for key in probability]
            sam=random.choices(probability,wght).pop()
            sample_spaces[sam]+=1

    sample_spaces={key:value/n for key,value in sample_spaces.items()}

    if round(sum(sample_spaces.values()),5) !=1:
        print(f'ERROR! probabilities add up to {sum(model_trans.values())}')

    else:
        print(f'sum of sample_pagerank values: {round(sum(sample_spaces.values()),10)}')
    return sample_spaces




def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    itr_pgr={}

    page_num=len(corpus)

    for page in corpus:
        itr_pgr[page]=1/page_num

    chng=1
    iter=1
   
    while chng>=0.001:
        chng=0
        prev_state=itr_pgr.copy()

        for page in itr_pgr:
            old_pg=[link for link in corpus if page in corpus[link]]

            first_pg=((1-damping_factor)/page_num)

            scnd=[]
            if len(old_pg)!=0:
                for k in old_pg:
                    no_of_links=len(corpus[k])
                    value=prev_state[k]/no_of_links
                    scnd.append(value)
            
            scnd=sum(scnd)
            itr_pgr[page]=first_pg+ (damping_factor * scnd)

            new_chng=abs(itr_pgr[page]-prev_state[page])

            if chng<new_chng:
                chng=new_chng
        iter+=1

    sumdict=sum(itr_pgr.values())
    itr_pgr={key:value/sumdict for key,value in itr_pgr.items()}

    print(f'pagerank results from {iter} iterations.')

    print(f'sum is : {round(sum(itr_pgr.values()),10)}')
    return itr_pgr

if __name__ == "__main__":
    main()
