#!/usr/bin/env python3 
"""
Product Hunt Scraper (Metadata Only)
This script downloads a sitemap .gz file from Product Hunt, unzips it, reads the sitemap.xml contained in the unzipped, extracts product URLs, and saves product metadata.
"""

import gzip
import shutil
import xml.etree.ElementTree as ET
import os
import sys
import asyncio
import requests
import json
import urllib.parse
import datetime
import logging
import nest_asyncio
import subprocess
import sqlite3
import hashlib
import tldextract
import numpy as np 
import psutil
import gc

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine 
from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
# TODO for embedding pip install sentence-transformers numpy scipy

# set up how the script will log output messages 
logging.basicConfig(
    level=logging.INFO, # only see logs like INFO, WARNING, ERROR, CRITICAL, and all messages with levels below INFO will be ignored
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"), # writes logs to a file named "scraper.log"
        logging.StreamHandler() # streams the logs to the terminal (stdout)
    ]
)
logger = logging.getLogger() # grabs the root logger instance so it can be used throughout the script. 

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # absolute path to the directory where the script is located 
SITEMAP_PATH = os.path.join(BASE_DIR, "product_imported_sitemap.xml") 
METADATA_OUTPUT_DIR = os.path.join(BASE_DIR, "product_hunt_metadata") # defines where to save the metadata JSON files 
OUTPUT_URLS_FILE = os.path.join(BASE_DIR, "product_urls.txt") # a raw list of all URLs extracted from the sitemap
MAX_CONCURRENT = 4  # Default concurrency 
MAX_URLS = 10  # 0 means crawl all URLs in the sitemap.
SITEMAP_URL = "https://www.producthunt.com/sitemaps_v3/product_imported_sitemap.xml.gz"
SITEMAP_GZ_PATH = os.path.join(BASE_DIR, "product_imported_sitemap.xml.gz")
MODEL = None

# Create output directories
os.makedirs(METADATA_OUTPUT_DIR, exist_ok=True)


# initialize SentenceTransformer model 
def get_sentence_transformer():
    global MODEL 
    if MODEL is None:
        MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return MODEL 


# create sqlite DB to track processed URLs
def initialize_database():
    db_path = os.path.join(BASE_DIR, "processed_urls.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # create table for sanity test 
    # url -> the product hunt listing page for that tool (want to ensure that this is unique so we don't process same page twice)
    # product_url -> home page of the tool 
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

    initialize_embedding_table(conn, cursor)
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


def initialize_embedding_table(conn, cursor):
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS product_embeddings (
        url TEXT PRIMARY KEY,
        embedding BLOB,
        FOREIGN KEY(url) REFERENCES processed_urls(url)
    )
    ''')
    conn.commit()


# not sure about correct threshold value 
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


# check if Product Hunt URL has been processed
def is_url_processed(cursor, url):
    cursor.execute("SELECT url FROM processed_urls WHERE url = ?", (url,)) # searches for a row(url,) where the url matches 
    return cursor.fetchone() is not None # True if URL exists in table | False if URL not found


# mark url as processed
def mark_url_processed(conn, cursor, url, product_info):
    now = datetime.datetime.now().isoformat() # current timestamp 
    product_url = product_info.get('product_url', '')
    title = product_info.get('title', '')
    normalized_name = normalize_product_name(title)
    root_domain = extract_root_domain(product_url)

    # might have to change this, instead of REPLACE INTO, just skip if it already exists 
    cursor.execute(
        "INSERT OR REPLACE INTO processed_urls VALUES (?, ?, ?, ?, ?, ?)",
        (url, product_url, title, normalized_name, root_domain, now)
    )
    conn.commit()


# Function to download and extract sitemap
def download_and_extract_sitemap():
    try:
        logger.info(f"Downloading sitemap from {SITEMAP_URL}")
        
        # Clean up any existing files -> 1. prevents accidental reuse of stale files 2. if the download fails halfway, it won't corrupt future runs since we're automating.
        if os.path.exists(SITEMAP_GZ_PATH):
            os.remove(SITEMAP_GZ_PATH)
        if os.path.exists(SITEMAP_PATH):
            os.remove(SITEMAP_PATH)
        
        # Download using curl commands 
        curl_cmd = ["curl", "-L", "-o", SITEMAP_GZ_PATH, SITEMAP_URL] # Another approach using lowercase -o in case they change their .gz url
        result = subprocess.run(curl_cmd, check=True, capture_output=True, text=True) # Execute the curl command 
        logger.info(f"Downloaded sitemap.xml.gz with curl")
        
        # Extract using gzip
        gzip_cmd = ["gzip", "-d", "-f", SITEMAP_GZ_PATH] # unzip
        result = subprocess.run(gzip_cmd, check=True, capture_output=True, text=True) # execute unzip command
        logger.info(f"Extracted sitemap.xml with gzip")
        
        # to debug can do print(result.stdout)

        # Check if extraction produced the expected file
        if not os.path.exists(SITEMAP_PATH):
            raise Exception("Extraction completed but sitemap.xml not found")
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading or extracting sitemap: {e}")
        return False
    

# extract product information from Product Hunt
def extract_product_info(producthunt_url, html_content):
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
                # Remove any Product Hunt tracking parameters like ?ref=producthunt so we can have a clean base URL 
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
            'description': "",
            'producthunt_url': producthunt_url
        }


# Define the crawling function (metadata only)
async def crawl_parallel(urls, conn, cursor, max_concurrent=4):
    logger.info("\nCrawling Product Hunt Pages for Metadata ===")

    logger.info(f"Setting up browser config with headless={True}")
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", 
        "--disable-dev-shm-usage", 
        "--no-sandbox",
        "--disable-extensions",
        "--disable-translate",
        "--disable-web-security",
        "--disable-background-networking",
        "--disable-default-apps",
        "--single-process",  # Important - use a single process
        "--no-zygote",
        "--disable-setuid-sandbox",
        "--mute-audio"],
    )
    
    logger.info("Setting up crawler config with cache mode BYPASS")
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Start the crawler
    logger.info("Initializing crawler...")
    crawler = AsyncWebCrawler(config=browser_config)
    
    logger.info("Starting crawler...")
    try:
        await crawler.start()
        logger.info("Crawler successfully started")
    except Exception as e:
        logger.critical(f"failed to start crawler: {e}")
        raise

    try:    
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}: {batch}")
            
            # for each URL in the batch, schedule an async crawl (yet to be executed)
            logger.info("Scheduling crawler tasks...")
            tasks = []
            for url in batch:
                logger.info(f"Scheduling crawl for: {url}")
                task = crawler.arun(url=url, config=crawl_config)
                tasks.append((url, task))
            
            logger.info(f"Gathering results for {len(tasks)} tasks...")
            logger.info("About to execute asyncio.gather()...")

            # logs detailed system resource usage
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                open_files = len(process.open_files())
                connections = len(process.connections())
                threads = process.num_threads()
                
                logger.info(f"System stats before gather: "
                            f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB, "
                            f"CPU: {cpu_percent}%, "
                            f"Open files: {open_files}, "
                            f"Connections: {connections}, "
                            f"Threads: {threads}")
                
                # Log available system memory
                system_memory = psutil.virtual_memory()
                logger.info(f"System memory: {system_memory.available / (1024 * 1024):.2f} MB available out of {system_memory.total / (1024 * 1024):.2f} MB")
            
            except ImportError:
                logger.info("psutil not available for system monitoring")
            except Exception as e:
                logger.warning(f"Failed to get system stats: {e}")

            try:
                # Check Chrome processes
                chrome_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                    if 'chrome' in proc.info['name'].lower():
                        chrome_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'memory': proc.info['memory_info'].rss / (1024 * 1024) if proc.info['memory_info'] else 0
                        })
                
                if chrome_processes:
                    logger.info(f"Chrome processes: {len(chrome_processes)}")
                    for proc in chrome_processes:
                        logger.info(f"Chrome process: PID={proc['pid']}, Name={proc['name']}, Memory={proc['memory']:.2f} MB")
                else:
                    logger.info("No Chrome processes found")
            except Exception as e:
                logger.warning(f"Failed to check Chrome processes: {e}")

            try:
                # Use gather to run multiple async tasks in parallel 
                results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)
                logger.info(f"Gathered {len(results)} results")
            except Exception as e:
                logger.critical(f"error during gather: {e}")
                results = [Exception(f"Gather failed: {e}") for _ in tasks]
                
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent(interval=0.1)
                open_files = len(process.open_files())
                connections = len(process.connections())
                threads = process.num_threads()
                
                logger.info(f"System stats after gather: "
                            f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB, "
                            f"CPU: {cpu_percent}%, "
                            f"Open files: {open_files}, "
                            f"Connections: {connections}, "
                            f"Threads: {threads}")
                
                # Log available system memory
                system_memory = psutil.virtual_memory()
                logger.info(f"System memory: {system_memory.available / (1024 * 1024):.2f} MB available out of {system_memory.total / (1024 * 1024):.2f} MB")
            
            except ImportError:
                logger.info("psutil not available for system monitoring")
            except Exception as e:
                logger.warning(f"Failed to get system stats: {e}")

            # iterate over each crawl task and their results in parallel
            logger.info("Processing crawl results...")
            for idx, ((url, _), result) in enumerate(zip((t for t in tasks), results)):
                logger.info(f"Processing result {idx+1}/{len(results)} for URL: {url}")
                
                if isinstance(result, Exception):
                    logger.error(f"Error scraping Product Hunt page {url}: {result}")
                elif result.success:
                    logger.info(f"Successfully crawled: {url}")
                    try:
                        # Extract product info from Product Hunt page
                        logger.info(f"Extracting product info from: {url}")
                        product_info = extract_product_info(url, result.html)
                        logger.info(f"Extracted product info: {product_info['title']}")

                        # check similar products using embeddings
                        if product_info['title'] != "Product Page":  # Only if we have a proper title
                            logger.info(f"Generating embedding for: {product_info['title']}")
                            try:
                                product_embedding = get_product_embedding(product_info)
                                logger.info(f"Generated embedding, checking for similar products...")
                                is_similar, similar_product, similar_url, similarity = check_similar_products_by_embedding(cursor, product_embedding)
                            except Exception as e:
                                logger.error(f"Error during embedding generation/comparison: {e}")
                                is_similar = False
                            
                            if is_similar:
                                logger.info(f"Found similar product: {similar_product[0]} ({similarity:.2f} similarity)")
                                logger.info(f"Skipping '{product_info['title']}' as it appears to be duplicate of '{similar_product[0]}'")
                                
                                # Mark as processed but note the duplicate
                                product_info['duplicate_of'] = similar_url
                                product_info['similarity_score'] = float(similarity)
                                logger.info(f"Marking as processed (duplicate): {url}")
                                mark_url_processed(conn, cursor, url, product_info)
                                continue
                            else:
                                # Store this product's embedding for future comparison
                                logger.info(f"Storing embedding for: {product_info['title']}")
                                try:
                                    store_product_embedding(conn, cursor, url, product_embedding)
                                    logger.info(f"Successfully stored embedding for: {product_info['title']}")
                                except Exception as e:
                                    logger.error(f"Failed to store embedding: {e}")

                        # Mark this URL as processed in the database
                        logger.info(f"Marking as processed: {url}")
                        mark_url_processed(conn, cursor, url, product_info)
                        logger.info(f"Successfully marked as processed: {url}")

                        # Save the metadata as JSON
                        try:
                            metadata_filename = product_info['title'].lower().replace(' ', '-') + ".json"
                            metadata_filepath = os.path.join(METADATA_OUTPUT_DIR, metadata_filename)
                            logger.info(f"Saving metadata to: {metadata_filepath}")
                            
                            # Add timestamp to metadata
                            product_info['crawl_timestamp'] = datetime.datetime.now().isoformat()
                            
                            with open(metadata_filepath, "w", encoding="utf-8") as file:
                                json.dump(product_info, file, indent=2)
                            logger.info(f"Successfully saved metadata: {metadata_filename}")
                        except Exception as e:
                            logger.error(f"Failed to save metadata: {e}")
                    except Exception as e:
                        logger.error(f"Error processing Product Hunt page {url}: {e}")
                        logger.exception("Exception details:")
                else:
                    logger.warning(f"Failed to scrape Product Hunt page {url}")
            
            if (i + max_concurrent) < len(urls):
                logger.info("Restarting crawler to free up memory...")
                try:
                    await crawler.close()
                    logger.info("Crawler closed")
                    # Force garbage collection
                    gc.collect()
                    logger.info("Garbage collection completed")
                    # Restart crawler
                    crawler = AsyncWebCrawler(config=browser_config)
                    await crawler.start()
                    logger.info("Crawler restarted")
                except Exception as e:
                    logger.error(f"Error restarting crawler: {e}")
                    
            # Also add a short delay to avoid rate limiting
            logger.info("Adding delay between batches to reduce load and avoid rate limiting")
            await asyncio.sleep(5)

    except Exception as e:
        logger.critical(f"error in crawl_parallel: {e}")
        logger.exception("Exception details:")
        raise

    finally:
        logger.info("\nClosing crawler...")
        try:
            await crawler.close()
            logger.info("Crawler successfully closed")
        except Exception as e:
            logger.error(f"Error closing crawler: {e}")


async def main():
    # Download and extract sitemap 
    logger.info("Starting Product Hunt metadata scraper")

    # Initialize the database connection
    conn, cursor = initialize_database()
    logger.info("Database initialized for tracking processed URLs")

    try:
        # Download and extract the sitemap 
        if not download_and_extract_sitemap():
            logger.error("Failed to download or extract sitemap. Exiting.")
            return 1
    
        # Parse the XML and extract product URLs
        logger.info(f"Parsing sitemap from {SITEMAP_PATH}")
        
        tree = ET.parse(SITEMAP_PATH)
        root = tree.getroot()

        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Extract all product URLs
        urls = []
        try:
            urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
            
            # If no URLs found, try without namespace
            if not urls:
                urls = [loc.text for loc in root.findall('.//loc')]
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
            logger.info("Trying alternative parsing method...")

            # generic XML parsing method if previous fails 
            for child in root:
                for element in child:
                    if element.tag.endswith('loc'):
                        urls.append(element.text)
        
        # Save URLs to a text file
        with open(OUTPUT_URLS_FILE, "w") as f:
            for url in urls:
                f.write(url + "\n")
        
        logger.info(f"Found {len(urls)} product URLs")
        logger.info(f"Product URLs saved to: {OUTPUT_URLS_FILE}")
        
        # Filter URLs that have already been processed
        filtered_urls = []
        skipped_count = 0

        for url in urls:
            if is_url_processed(cursor, url):
                skipped_count += 1
            else:
                filtered_urls.append(url)

        logger.info(f"Found {len(urls)} URLs, {skipped_count} already processed, {len(filtered_urls)} new")

        urls_to_process = filtered_urls[:MAX_URLS] if MAX_URLS > 0 else filtered_urls
        logger.info(f"Processing {len(urls_to_process)} new URLs")
        
        logger.info(f"Starting crawler with max_concurrent={MAX_CONCURRENT}")
        try:
            await crawl_parallel(urls_to_process, conn, cursor, max_concurrent=MAX_CONCURRENT)
            logger.info("Crawler finished successfully")
        except Exception as e:
            logger.critical(f"Crawler failed with error: {e}")
            logger.exception("Crawler exception details:")
            return 1
            
        logger.info(f"\nAll metadata files saved in: {METADATA_OUTPUT_DIR}")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # time the scraper finished + how many URLs were processed 
        with open(os.path.join(BASE_DIR, f"last_run_{timestamp}.txt"), "w") as f:
            f.write(f"Scraper ran successfully at {datetime.datetime.now().isoformat()}\n")
            f.write(f"Processed {len(urls_to_process)} URLs\n")
            f.write(f"Skipped {skipped_count} previously processed URLs\n")
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.exception("Exception details:")
        return 1
    
    finally:
        conn.close()
        logger.info("Database connection closed")

    return 0

if __name__ == "__main__":
    # Apply nest_asyncio to allow nested event loops (important for asyncio in scripts)
    nest_asyncio.apply()
    
    # Run the async main function
    loop = asyncio.get_event_loop()
    exit_code = loop.run_until_complete(main())
    
    # Exit with appropriate code
    sys.exit(exit_code)