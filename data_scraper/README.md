# Get It Started 

Example of how to run the pull request/repository scraper (*github_scraper.py*):

```python
scraper = GitHubScraper(tokens = your_tokens, batch_size = 30)

start_date = "2025-05-01T00:00:00Z"
end_date = "2025-05-07T00:00:00Z"

filter = "head:copilot/"

for contribution in scraper.scrape_prs(start_date = start_date, end_date = end_date, filter = filter, total = 5000)
    print(contribution)
```

```tokens```: your GitHub tokens, you can pass more than one, the token manager will cycle through them whenever limits are hit 

```batch_size```: number of pull requests to scraper per page 

```start_date```: start date of the scraping interval 

```end_date```: end date of the scraping interval

```filter```: signal used to filter pull requests according to each agent (all signals are presented in the paper)

```total```: total amount of pull requests to be scraped

Example of how to run the commit/comment/review/issue scraper (*github_scraper_rest.py*) based on the obtained pull request IDs: 

```python
# First load the pull request dataset as shown on the HuggingFace page, then pass it to the scrape_prs method.
...
pull_request_ids = dataset['train']['id']

for contribution in scraper.scrape_prs(pr_ids = pull_request_ids)
    print(contribution)
```

```pr_ids```: pull request IDs 

Example of how to run the pull request/repository scraper for humans (*github_scraper_human.py*): 

```python
scraper = GitHubScraper(tokens = your_tokens, batch_size = 30)

start_date = "2025-05-01T00:00:00Z"
end_date = "2025-05-07T00:00:00Z"

filter = "-head:copilot/"

for contribution in scraper.scrape_prs(start_date = start_date, end_date = end_date, filter = filter, total = 5000, interval_span = 3)
    print(contribution)
```

```filter```: we prepend "-" to the signal to exclude copilot pull requests. This can be chanined with all the other signals presented in the paper

```interval_span```: length of the intervals in which a day is split. 3 means that a day is split into 3-hour intervals
