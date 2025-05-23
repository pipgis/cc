import requests
from bs4 import BeautifulSoup
import feedparser
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

def fetch_url_content(url: str) -> dict:
    """
    Fetches content from a single URL, parses HTML, and extracts news data.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for URL: {url}", exc_info=True)
        return {'error': f"Request failed: {e}", 'source_url': url}

    soup = BeautifulSoup(response.content, 'html.parser')

    title = None
    # Try to get title from <title> tag
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    # Fallback to h1 if title is not found or empty
    if not title:
        h1_tag = soup.find('h1')
        if h1_tag:
            title = h1_tag.get_text(separator=' ', strip=True)
    # Further fallback for other common headline tags can be added here

    summary = None
    # Try to get summary from <meta name="description">
    meta_description = soup.find('meta', attrs={'name': 'description'})
    if meta_description and meta_description.get('content'):
        summary = meta_description.get('content').strip()
    # Fallback to finding the first few <p> tags if no meta description
    if not summary:
        first_p = soup.find('p')
        if first_p:
            summary = first_p.get_text(separator=' ', strip=True)
            # Potentially concatenate a few paragraphs or limit length
            # summary = ' '.join([p.get_text(strip=True) for p in soup.find_all('p', limit=2)])

    published_date_iso = None
    # Try to get publication date from <meta property="article:published_time">
    meta_pub_time = soup.find('meta', attrs={'property': 'article:published_time'})
    if meta_pub_time and meta_pub_time.get('content'):
        try:
            # Content is usually in ISO format e.g., "2023-10-27T10:30:00Z"
            published_date_iso = datetime.fromisoformat(meta_pub_time.get('content').replace('Z', '+00:00')).isoformat()
        except ValueError as ve:
            logger.debug(f"Could not parse date from meta tag for {url}: {meta_pub_time.get('content')}", exc_info=True)
            pass
    # Add more date extraction logic here (e.g., time tags, specific class names)

    # Basic cleaning of title and summary
    if title:
        title = re.sub(r'\s+', ' ', title).strip()
    if summary:
        summary = re.sub(r'\s+', ' ', summary).strip()


    if not title and not summary: # If we couldn't find a title or summary, it's likely not a useful article page
        logger.warning(f"Could not extract meaningful content (title/summary missing) for URL: {url}")
        return {'error': 'Could not extract meaningful content (title/summary missing)', 'source_url': url}

    logger.info(f"Successfully fetched and parsed content from URL: {url}")
    return {
        'title': title,
        'summary': summary,
        'source_url': url,
        'published_date': published_date_iso,
        'error': None
    }

def fetch_rss_feed(feed_url: str) -> list:
    """
    Fetches and parses an RSS feed, extracting news items.
    """
    try:
        parsed_feed = feedparser.parse(feed_url)
    except Exception as e: # feedparser can raise various errors
        logger.error(f"Failed to parse RSS feed: {feed_url}", exc_info=True)
        return [{'error': f"Failed to parse RSS feed: {e}", 'source_url': feed_url}]

    if parsed_feed.bozo:
        bozo_exception_message = str(parsed_feed.bozo_exception)
        logger.warning(f"RSS feed '{feed_url}' may be ill-formed. Bozo exception: {bozo_exception_message}")
        if not parsed_feed.entries:
            logger.error(f"RSS feed '{feed_url}' is not well-formed and no entries found. Bozo exception: {bozo_exception_message}")
            return [{'error': f"RSS feed is not well-formed and no entries found: {bozo_exception_message}", 'source_url': feed_url}]

    news_items = []
    logger.info(f"Processing {len(parsed_feed.entries)} entries from RSS feed: {feed_url}")
    for entry_index, entry in enumerate(parsed_feed.entries):
        title = entry.get('title')
        link = entry.get('link')
        summary = entry.get('summary') or entry.get('description') # RSS feeds can use either

        published_date_iso = None
        published_time_struct = entry.get('published_parsed') or entry.get('updated_parsed')
        if published_time_struct:
            try:
                published_date_iso = datetime(*published_time_struct[:6]).isoformat()
            except TypeError: # If published_time_struct is None or not a valid time tuple
                logger.debug(f"Invalid date format in RSS entry {entry_index} for feed {feed_url}.", exc_info=True)
                pass # Keep published_date_iso as None

        # Basic cleaning
        if title:
            title = re.sub(r'\s+', ' ', BeautifulSoup(title, "html.parser").get_text(separator=' ', strip=True))
        if summary:
            summary = re.sub(r'\s+', ' ', BeautifulSoup(summary, "html.parser").get_text(separator=' ', strip=True))


        if link: # A link is essential for an RSS item to be useful
            news_items.append({
                'title': title,
                'summary': summary,
                'source_url': link,
                'published_date': published_date_iso,
                'error': None
            })
    return news_items

def fetch_news(sources: list[str]) -> list[dict]:
    """
    Fetches news from a list of URLs or RSS feed URLs.
    Differentiates source types and calls appropriate fetchers.
    Deduplicates results.
    """
    all_news_items = []
    processed_urls = set()
    logger.info(f"Starting news fetching process for {len(sources)} sources.")

    for source in sources:
        # Simple heuristic for RSS: ends with .xml, .rss, or contains "feed" or "rss" in path
        # More robust detection might involve trying to parse as feed and falling back.
        if source.endswith(('.xml', '.rss')) or "feed" in source.lower() or "rss" in source.lower():
            logger.info(f"Fetching RSS feed: {source}")
            rss_items = fetch_rss_feed(source)
            for item in rss_items:
                if item.get('error'):
                    logger.warning(f"Error processing RSS item from {source}: {item['error']}")
                    all_news_items.append(item) # Keep error items for now, could filter later
                elif item.get('source_url') and item['source_url'] not in processed_urls:
                    all_news_items.append(item)
                    processed_urls.add(item['source_url'])
                elif item.get('title') and item.get('title') not in processed_urls: # Fallback to title if URL is missing or duplicated
                    logger.debug(f"Adding RSS item by title (URL missing or duplicate): {item.get('title')}")
                    all_news_items.append(item)
                    processed_urls.add(item['title'])
                else:
                    logger.debug(f"Skipping duplicate or invalid RSS item: {item.get('title') or item.get('source_url')}")


        else:
            logger.info(f"Fetching URL content: {source}")
            if source not in processed_urls:
                item = fetch_url_content(source)
                if item.get('error'):
                     logger.warning(f"Error processing URL {source}: {item['error']}")
                all_news_items.append(item) # Add item even if there's an error to see what went wrong
                if item.get('source_url'): # Should always be there
                    processed_urls.add(item['source_url'])
            else:
                logger.info(f"Skipping already processed URL: {source}")

    logger.info(f"Collected {len(all_news_items)} items before deduplication.")
    # Deduplication (a more robust one after collection)
    # First pass was to avoid re-fetching. This pass is to ensure uniqueness in the final list.
    unique_items_dict = {}
    final_news_list = []

    for item in all_news_items:
        if item.get('error'): # Keep error items
            final_news_list.append(item)
            continue

        # Prioritize URL for uniqueness
        key = item.get('source_url')
        if key:
            if key not in unique_items_dict:
                unique_items_dict[key] = item
        else: # If no URL, use title (less reliable)
            key_title = item.get('title')
            if key_title and key_title not in unique_items_dict:
                 unique_items_dict[key_title] = item
            else:
                # If no URL and title is also missing or duplicate, we might still add it
                # or decide to discard it. For now, add if it has some content.
                if item.get('summary'): # Check if there's at least some content
                    final_news_list.append(item)


    final_news_list.extend(list(unique_items_dict.values()))
    logger.info(f"Returning {len(final_news_list)} unique news items after deduplication.")
    return final_news_list

if __name__ == '__main__':
    # Basic setup for testing this module directly
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    sample_sources = [
        "http://rss.cnn.com/rss/cnn_topstories.rss",
        "https://www.bbc.com/news", # General news site, likely hard to parse well with current simple logic
        "https://feeds.arstechnica.com/arstechnica/index", # Another RSS
        # "https://www.theverge.com/" # Example of a site that might be harder
        # Add a non-existent URL or a URL that might cause timeout for error testing
        "http://thisdomainprobablydoesnotexist12345.com",
        "https://httpstat.us/503" # Service unavailable
    ]

    logger.info("Fetching news from sample sources for direct module test...")
    news_items = fetch_news(sample_sources)
    logger.info(f"\nFetched {len(news_items)} items (including potential errors):")
    for i, item in enumerate(news_items):
        logger.info(f"\n--- Item {i+1} ---")
        logger.info(f"  Title: {item.get('title')}")
        logger.info(f"  Source: {item.get('source_url')}")
        logger.info(f"  Published: {item.get('published_date')}")
        logger.info(f"  Summary: {item.get('summary', 'N/A')[:100]}...") # Print first 100 chars of summary
        if item.get('error'):
            logger.error(f"  Error: {item.get('error')}")

    # Example of fetching a single URL directly (for testing fetch_url_content)
    logger.info("\n--- Testing single URL fetch ---")
    single_url_item = fetch_url_content("https://www.reuters.com/world/middle-east/israeli-tanks-push-deeper-rafah-residents-say-2024-05-29/")
    logger.info(f"  Title: {single_url_item.get('title')}")
    logger.info(f"  Source: {single_url_item.get('source_url')}")
    logger.info(f"  Published: {single_url_item.get('published_date')}")
    logger.info(f"  Summary: {single_url_item.get('summary', 'N/A')[:100]}...")
    if single_url_item.get('error'):
        logger.error(f"  Error: {single_url_item.get('error')}")

    # Example of fetching a single RSS feed directly
    logger.info("\n--- Testing single RSS fetch ---")
    single_rss_items = fetch_rss_feed("http://rss.cnn.com/rss/cnn_topstories.rss")
    logger.info(f"Fetched {len(single_rss_items)} items from single RSS:")
    for i, item in enumerate(single_rss_items[:2]): # Print first 2 items
        logger.info(f"  --- RSS Item {i+1} ---")
        logger.info(f"    Title: {item.get('title')}")
        logger.info(f"    Source: {item.get('source_url')}")
        logger.info(f"    Published: {item.get('published_date')}")
        logger.info(f"    Summary: {item.get('summary', 'N/A')[:100]}...")
        if item.get('error'):
            logger.error(f"    Error: {item.get('error')}")
