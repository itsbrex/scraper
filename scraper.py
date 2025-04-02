#!/usr/bin/env python3 
"""
Product Hunt Metadata Crawler for GitHub Actions
Automatically scrapes Product Hunt to get product information and metadata
Runs on a schedule with deduplication and automatic sitemap download
"""

import gzip
import shutil
import xml.etree.ElementTree as ET
import os
import sys
import asyncio
import requests
import json
import time
import random
import subprocess
import sqlite3
import tldextract
import numpy as np
import gc
import datetime
import logging
import urllib.parse
from bs4 import BeautifulSoup
import re
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
import nest_asyncio
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Setup directories - adjusted for GitHub Actions environment
BASE_DIR = os.getcwd()
os.makedirs(os.path.join(BASE_DIR, "sitemaps"), exist_ok=True)
METADATA_OUTPUT_DIR = os.path.join(BASE_DIR, "product_hunt_metadata")
os.makedirs(METADATA_OUTPUT_DIR, exist_ok=True)
OUTPUT_URLS_FILE = os.path.join(BASE_DIR, "product_urls.txt")
SITEMAP_URL = "https://www.producthunt.com/sitemaps_v3/product_about_sitemap1.xml.gz"
SITEMAP_GZ_PATH = os.path.join(BASE_DIR, "product_about_sitemap1.xml.gz")
SITEMAP_PATH = os.path.join(BASE_DIR, "product_about_sitemap1.xml")
MODEL = None

# Number of URLs to process in a single run (0 = all)
MAX_URLS = 5  # Limit to 100 per run for GitHub Actions
MAX_CONCURRENT = 5  # Default concurrency for GitHub Actions

# Function to download and extract sitemap
def download_and_extract_sitemap():
    """
    Download and extract the sitemap file automatically
    """
    try:
        logger.info(f"Downloading sitemap from {SITEMAP_URL}")
        
        # Clean up any existing files
        if os.path.exists(SITEMAP_GZ_PATH):
            os.remove(SITEMAP_GZ_PATH)
        if os.path.exists(SITEMAP_PATH):
            os.remove(SITEMAP_PATH)
        
        # Download the sitemap file
        try:
            # First try using curl for reliability with redirects
            curl_cmd = ["curl", "-L", "--connect-timeout", "30", "-m", "120", "-o", SITEMAP_GZ_PATH, SITEMAP_URL]
            result = subprocess.run(curl_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Downloaded sitemap.xml.gz with curl")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            # Fall back to requests if curl fails or isn't available
            logger.info(f"Curl failed or not available, using requests instead: {e}")
            response = requests.get(SITEMAP_URL, stream=True, timeout=60)
            response.raise_for_status()
            with open(SITEMAP_GZ_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Downloaded sitemap.xml.gz with requests")
        
        # Extract the gzip file
        try:
            # First try using gzip command
            gzip_cmd = ["gzip", "-d", "-f", SITEMAP_GZ_PATH]
            result = subprocess.run(gzip_cmd, check=True, capture_output=True, text=True)
            logger.info(f"Extracted sitemap.xml with gzip command")
            
            # Rename the extracted file if needed
            if os.path.exists(SITEMAP_GZ_PATH[:-3]):  # Removes .gz extension
                shutil.move(SITEMAP_GZ_PATH[:-3], SITEMAP_PATH)
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            # Fall back to Python's gzip module
            logger.info(f"Gzip command failed or not available, using Python's gzip module: {e}")
            with gzip.open(SITEMAP_GZ_PATH, 'rb') as f_in:
                with open(SITEMAP_PATH, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            logger.info(f"Extracted sitemap.xml with Python's gzip module")
        
        # Check if extraction produced the expected file
        if not os.path.exists(SITEMAP_PATH):
            raise Exception("Extraction completed but sitemap.xml not found")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading or extracting sitemap: {e}")
        return False


# Initialize SentenceTransformer model
def get_sentence_transformer():
    global MODEL 
    if MODEL is None:
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return MODEL

# Database functions for tracking processed URLs and embeddings
def initialize_database():
    db_path = os.path.join(BASE_DIR, "processed_urls.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table for processed URLs
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_urls (
        url TEXT PRIMARY KEY,
        product_url TEXT,
        title TEXT,
        normalized_name TEXT,
        root_domain TEXT,
        first_seen TIMESTAMP
    )
    ''')

    # Add indices for faster lookups
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_product_url ON processed_urls(product_url)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_normalized_name ON processed_urls(normalized_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_root_domain ON processed_urls(root_domain)')

    # Initialize embeddings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_embeddings (
        url TEXT PRIMARY KEY,
        embedding BLOB,
        FOREIGN KEY(url) REFERENCES processed_urls(url)
    )
    ''')
    
    conn.commit()
    return conn, cursor

def normalize_product_name(name):
    """Create a normalized version of product name for matching"""
    if not name:
        return ""
    # Remove spaces, lowercase, strip special chars
    return ''.join(c.lower() for c in name if c.isalnum())

def extract_root_domain(url):
    if not url:
        return ""
    try:
        ext = tldextract.extract(url)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
        return ext.domain
    except Exception as e:
        logger.error(f"Failed to parse URL: {url} — {e}")
        return ""

def get_product_embedding(product_info):
    # Combine product info into a descriptive text
    description = f"{product_info.get('title', '')} {product_info.get('description', '')}"
    
    # Generate embedding
    model = get_sentence_transformer()
    embedding = model.encode(description)
    return embedding

def check_similar_products_by_embedding(cursor, product_embedding, threshold=0.85):
    # Get all existing embeddings
    cursor.execute("SELECT url, embedding FROM product_embeddings")
    results = cursor.fetchall()
    
    for url, stored_embedding_blob in results:
        if stored_embedding_blob is None:
            continue
            
        # Convert blob back to numpy array
        stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float32)
        # Calculate similarity
        similarity = 1 - cosine(product_embedding, stored_embedding)
        
        logger.debug(f"Similarity with {url}: {similarity}")
        
        if similarity > threshold:
            # Found similar product
            cursor.execute("SELECT title, product_url FROM processed_urls WHERE url = ?", (url,))
            similar_product = cursor.fetchone()
            return True, similar_product, url, similarity
    
    return False, None, None, 0.0

def store_product_embedding(conn, cursor, url, embedding):
    """Store product embedding in the database"""
    embedding_blob = embedding.astype(np.float32).tobytes()
    cursor.execute("INSERT OR REPLACE INTO product_embeddings VALUES (?, ?)", (url, embedding_blob))
    conn.commit()

# Check if Product Hunt URL has been processed
def is_url_processed(cursor, url):
    cursor.execute("SELECT url FROM processed_urls WHERE url = ?", (url,))
    return cursor.fetchone() is not None

# Mark URL as processed
def mark_url_processed(conn, cursor, url, product_info):
    now = datetime.datetime.now().isoformat()
    product_url = product_info.get('product_url', '')
    title = product_info.get('title', '')
    normalized_name = normalize_product_name(title)
    root_domain = extract_root_domain(product_url)

    cursor.execute(
        "INSERT OR REPLACE INTO processed_urls VALUES (?, ?, ?, ?, ?, ?)",
        (url, product_url, title, normalized_name, root_domain, now)
    )
    conn.commit()

# Rate limiter class
class RateLimiter:
    """
    Rate limiter to control request frequency to domains
    """
    def __init__(self, default_rate=1.0, per_domain_rates=None):
        """
        Initialize the rate limiter
        
        Args:
            default_rate: Default requests per second (float)
            per_domain_rates: Dict mapping domains to their specific rates
        """
        self.default_rate = default_rate  # requests per second
        self.per_domain_rates = per_domain_rates or {}
        self.last_request_time = {}
        
    def get_delay_for_domain(self, domain):
        """
        Calculate how long to wait before the next request to this domain
        
        Args:
            domain: The domain name
            
        Returns:
            Delay in seconds (float)
        """
        current_time = time.time()
        domain_rate = self.per_domain_rates.get(domain, self.default_rate)
        
        # Time between requests in seconds
        time_between_requests = 1.0 / domain_rate
        
        # Add some jitter (±20%) to avoid predictable patterns
        jitter = random.uniform(0.8, 1.2)
        time_between_requests *= jitter
        
        last_request = self.last_request_time.get(domain, 0)
        elapsed = current_time - last_request
        
        if elapsed >= time_between_requests:
            # No need to wait
            self.last_request_time[domain] = current_time
            return 0
        else:
            # Calculate remaining wait time
            delay = time_between_requests - elapsed
            self.last_request_time[domain] = current_time + delay
            return delay
    
    async def wait_for_domain(self, domain):
        """
        Wait the appropriate time before making a request to this domain
        
        Args:
            domain: The domain name
        """
        delay = self.get_delay_for_domain(domain)
        if delay > 0:
            logger.info(f"⏱️ Rate limiting: waiting {delay:.2f}s before requesting {domain}")
            await asyncio.sleep(delay)

# Function to extract product information from Product Hunt
def extract_product_info(producthunt_url, html_content):
    """
    Extract basic product information from a Product Hunt page
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract product title
        title = "Product Page"
        h1_tags = soup.find_all('h1')
        if h1_tags:
            for h1 in h1_tags:
                if h1.text.strip():
                    title = h1.text.strip()
                    break

        # Extract the actual product website URL
        product_url = None
        for a in soup.find_all('a'):
            if a.text and 'Visit website' in a.text and a.get('href'):
                product_url = a.get('href')
                # Remove any Product Hunt tracking parameters
                if '?' in product_url:
                    base_url = product_url.split('?')[0]
                    product_url = base_url
                break

        # Extract product description - improved method
        description = None
        
        # Method 1: Look for section with h2 "SectionTitle" followed by div with description text
        section_elements = soup.find_all('section', {'class': lambda x: x and 'flex' in x and 'flex-col' in x and 'gap-2' in x})
        for section in section_elements:
            # Check if this section has a SectionTitle h2
            section_title = section.find('h2', {'data-sentry-component': 'SectionTitle'})
            if section_title:
                # Look for the description div that follows
                desc_div = section.find('div', {'class': ['text-16', 'text-secondary']})
                if desc_div and desc_div.text.strip():
                    # Ensure this is a substantial description (at least 30 chars)
                    desc_text = desc_div.text.strip()
                    if len(desc_text) > 30 and not desc_text.startswith('Coming soon') and '==' not in desc_text:
                        description = desc_text
                        break
        
        # Method 2: Find div with specific classes that looks like a description
        if not description:
            for div in soup.find_all('div', {'class': ['text-16', 'text-secondary']}):
                text = div.text.strip()
                # Filter out navigation elements and other non-description text
                if (len(text) > 50 and 
                    not text.startswith('Coming soon') and
                    'launches' not in text.lower() and
                    'community' not in text.lower() and
                    '==' not in text):
                    description = text
                    break
        
        # Method 3: Look for LegacyText components with descriptive content
        if not description:
            legacy_texts = soup.find_all('div', {'data-sentry-component': 'LegacyText'})
            for text_div in legacy_texts:
                # Check if this might be a description (longer text, not just a title)
                text = text_div.text.strip()
                if (len(text) > 50 and 
                    ('tool' in text.lower() or 'platform' in text.lower() or 
                    'service' in text.lower() or 'app' in text.lower())):
                    description = text
                    break

        return {
            'title': title,
            'product_url': product_url,
            'description': description,
            'producthunt_url': producthunt_url
        }

    except Exception as e:
        logger.error(f"Error extracting product info: {e}")
        return {
            'title': producthunt_url.split('/')[-1].replace('-', ' ').title(),
            'product_url': None,
            'description': None,
            'producthunt_url': producthunt_url
        }

# Define the crawling function - with deduplication and rate limiting
async def crawl_parallel(urls, conn, cursor, max_concurrent=3):
    logger.info("\n=== 🚀 Crawling Product Hunt Pages for Metadata ===")

    # Configure headless browser
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=[
            "--disable-gpu", 
            "--disable-dev-shm-usage", 
            "--no-sandbox",
            "--disable-extensions",
            "--disable-translate",
            "--disable-web-security",
            "--disable-background-networking",
            "--disable-default-apps",
            "--mute-audio"
        ],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Start the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    # Initialize rate limiter
    rate_limiter = RateLimiter(
        default_rate=1.0,  # 1 request per second as default
        per_domain_rates={
            "www.producthunt.com": 0.5,  # 1 request per 2 seconds for Product Hunt
        }
    )

    try:
        # Process URLs in batches
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}: {len(batch)} URLs")
            
            # Create tasks for the batch with rate limiting
            tasks = []
            for url in batch:
                # Skip if already processed
                if is_url_processed(cursor, url):
                    logger.info(f"Skipping already processed URL: {url}")
                    continue
                
                # Extract domain for rate limiting
                domain = urllib.parse.urlparse(url).netloc
                
                # Wait according to rate limits
                await rate_limiter.wait_for_domain(domain)
                
                # Add the task
                tasks.append((url, crawler.arun(url=url, config=crawl_config)))
            
            # Execute tasks and gather results
            results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)

            # Process results
            for (url, result) in zip((url for url, _ in tasks), results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error scraping Product Hunt page {url}: {result}")
                elif result.success:
                    try:
                        # Extract product info from Product Hunt page
                        product_info = extract_product_info(url, result.html)
                        
                        # Check for similar products using embeddings
                        if product_info['title'] != "Product Page" and product_info['description']:
                            try:
                                product_embedding = get_product_embedding(product_info)
                                is_similar, similar_product, similar_url, similarity = check_similar_products_by_embedding(cursor, product_embedding)
                                
                                if is_similar:
                                    logger.info(f"Found similar product: {similar_product[0]} ({similarity:.2f} similarity)")
                                    logger.info(f"Skipping '{product_info['title']}' as it appears to be duplicate of '{similar_product[0]}'")
                                    
                                    # Mark as processed but note the duplicate
                                    product_info['duplicate_of'] = similar_url
                                    product_info['similarity_score'] = float(similarity)
                                    mark_url_processed(conn, cursor, url, product_info)
                                    continue
                                else:
                                    # Store this product's embedding for future comparison
                                    logger.info(f"Storing embedding for: {product_info['title']}")
                                    store_product_embedding(conn, cursor, url, product_embedding)
                            except Exception as e:
                                logger.error(f"Error during embedding processing: {e}")
                        
                        # Print info about what we found
                        if product_info.get('product_url'):
                            logger.info(f"✅ Found website URL for {product_info['title']}: {product_info['product_url']}")
                        else:
                            logger.info(f"⚠️ No website URL found for {product_info['title']}")
                            
                        # Get description info for logging
                        desc_length = len(product_info.get('description', '')) if product_info.get('description') else 0
                        if desc_length > 0:
                            logger.info(f"✅ Found description ({desc_length} chars) for {product_info['title']}")
                        else:
                            logger.info(f"⚠️ No description found for {product_info['title']}")

                        # Mark this URL as processed in the database
                        mark_url_processed(conn, cursor, url, product_info)

                        # Add timestamp to metadata
                        product_info['crawl_timestamp'] = datetime.datetime.now().isoformat()

                        # Save the metadata as JSON (create safe filename from title)
                        safe_title = product_info['title'].lower()
                        safe_title = re.sub(r'[^\w\s-]', '', safe_title)  # Remove special chars
                        safe_title = re.sub(r'\s+', '-', safe_title)      # Replace spaces with hyphens
                        metadata_filename = safe_title + ".json"
                        metadata_filepath = os.path.join(METADATA_OUTPUT_DIR, metadata_filename)
                        
                        with open(metadata_filepath, "w", encoding="utf-8") as file:
                            json.dump(product_info, file, indent=2)
                        logger.info(f"✅ Saved metadata: {metadata_filename}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing Product Hunt page {url}: {e}")
                else:
                    logger.warning(f"⚠️ Failed to scrape Product Hunt page {url}")
            
            # Add a short delay between batches to avoid rate limiting
            await asyncio.sleep(5)

    finally:
        logger.info("\nClosing crawler...")
        await crawler.close()
        
    logger.info(f"\n✅ Completed metadata collection")
    return

async def main():
    logger.info("Starting Product Hunt metadata crawler")
    
    # Initialize the database connection
    conn, cursor = initialize_database()
    logger.info("Database initialized for tracking processed URLs")
    
    try:
        # Step 1: Download the sitemap file automatically
        logger.info("Attempting to download and extract sitemap...")
        if not download_and_extract_sitemap():
            logger.error("Failed to download or extract sitemap. Exiting.")
            return 1
        
        # Step 2: Parse the XML and extract product URLs
        logger.info("🔍 Parsing sitemap XML...")
        tree = ET.parse(SITEMAP_PATH)
        root = tree.getroot()
        
        # Define namespace (if needed)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract all product URLs
        urls = []
        try:
            # First try with namespace
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
            # If no URLs found, try without namespace
            if not urls:
                urls = [loc.text for loc in root.findall('.//loc')]
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            logger.info("Trying alternative parsing method...")
            # Fallback to more generic XML parsing
            for child in root:
                for element in child:
                    if element.tag.endswith('loc'):
                        urls.append(element.text)
        
        # Save URLs to a text file
        with open(OUTPUT_URLS_FILE, "w") as f:
            for url in urls:
                f.write(url + "\n")
        
        logger.info(f"✅ Found {len(urls)} product URLs")
        
        # Filter out already processed URLs
        filtered_urls = []
        skipped_count = 0
        
        for url in urls:
            if is_url_processed(cursor, url):
                skipped_count += 1
            else:
                filtered_urls.append(url)
        
        logger.info(f"Found {len(urls)} URLs in total")
        logger.info(f"Skipping {skipped_count} previously processed URLs")
        logger.info(f"Processing {len(filtered_urls)} new URLs")
        
        # For GitHub Actions, limit the number of URLs processed per run
        urls_to_process = filtered_urls[:MAX_URLS] if MAX_URLS > 0 else filtered_urls
        logger.info(f"✅ Will process {len(urls_to_process)} URLs in this run")
        
        # Run the crawler with the configured concurrency
        await crawl_parallel(urls_to_process, conn, cursor, max_concurrent=MAX_CONCURRENT)
        logger.info(f"✅ All metadata files saved in: {METADATA_OUTPUT_DIR}")
        
        # Create a summary of the run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(BASE_DIR, f"last_run_{timestamp}.txt"), "w") as f:
            f.write(f"Scraper ran successfully at {datetime.datetime.now().isoformat()}\n")
            f.write(f"Processed {len(urls_to_process)} URLs\n")
            f.write(f"Skipped {skipped_count} previously processed URLs\n")
            f.write(f"Total URLs in sitemap: {len(urls)}\n")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.exception("Exception details:")
        return 1
    
    finally:
        conn.close()
        logger.info("Database connection closed")
    
    return 0

# Run the main function
if __name__ == "__main__":
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    exit_code = loop.run_until_complete(main())
    sys.exit(exit_code)
