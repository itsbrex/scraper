#!/usr/bin/env python3
"""
GitHub Actions Artifact to MongoDB Transfer Script

This script:
1. Downloads the latest Product Hunt metadata artifact from GitHub Actions
2. Extracts all JSON files
3. Processes each file to extract specific fields
4. Uploads the data to a MongoDB collection
"""

import os
import sys
import json
import requests
import zipfile
import tempfile
import logging
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("copy_agents.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration
GITHUB_REPOSITORY = "trolex213/scraper"  # Your GitHub repository
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Set as environment variable for security
MONGODB_URI = os.environ.get("MONGODB_URI")
MONGODB_DB = os.environ.get("MONGODB_DB", "product_hunt_db")
MONGODB_COLLECTION = os.environ.get("MONGODB_COLLECTION", "products")

def get_latest_artifact_url():
    """Get the download URL for the latest product-hunt-metadata artifact"""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    # Get the latest workflow run
    runs_url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/runs"
    runs_response = requests.get(runs_url, headers=headers)
    runs_response.raise_for_status()
    
    workflows = runs_response.json().get("workflow_runs", [])
    if not workflows:
        raise Exception("No workflow runs found")
    
    # Find the latest successful run
    latest_successful_run = None
    for run in workflows:
        if run["status"] == "completed" and run["conclusion"] == "success":
            latest_successful_run = run
            break
    
    if not latest_successful_run:
        raise Exception("No successful workflow runs found")
    
    run_id = latest_successful_run["id"]
    logger.info(f"Found latest successful run: {run_id}")
    
    # Get artifacts from the run
    artifacts_url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/runs/{run_id}/artifacts"
    artifacts_response = requests.get(artifacts_url, headers=headers)
    artifacts_response.raise_for_status()
    
    artifacts = artifacts_response.json().get("artifacts", [])
    
    # Find the product-hunt-metadata artifact
    metadata_artifact = None
    for artifact in artifacts:
        if "product-hunt-metadata" in artifact["name"]:
            metadata_artifact = artifact
            break
    
    if not metadata_artifact:
        raise Exception("No product-hunt-metadata artifact found")
    
    logger.info(f"Found metadata artifact: {metadata_artifact['name']}")
    
    # Return the download URL
    return metadata_artifact["archive_download_url"], metadata_artifact["name"]

def download_and_extract_artifact(download_url, artifact_name):
    """Download and extract the artifact zip file"""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN environment variable not set")
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    logger.info(f"Downloading artifact: {artifact_name}")
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, f"{artifact_name}.zip")
    
    # Download the zip file
    response = requests.get(download_url, headers=headers, stream=True)
    response.raise_for_status()
    
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Downloaded artifact to: {zip_path}")
    
    # Extract the zip file
    extract_dir = os.path.join(temp_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    logger.info(f"Extracted artifact to: {extract_dir}")
    
    return extract_dir

def process_json_files(extract_dir):
    """Process each JSON file and extract the required fields"""
    products = []
    
    # Find all JSON files in the extract directory
    json_files = []
    for root, _, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract the required fields
            product = {
                "title": data.get("title"),
                "product_url": data.get("product_url"),
                "description": data.get("description"),
                "producthunt_url": data.get("producthunt_url")
            }
            
            # Add to products list
            products.append(product)
            
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
    
    logger.info(f"Processed {len(products)} products")
    
    return products

def upload_to_mongodb(products):
    """Upload the products to MongoDB"""
    if not products:
        logger.warning("No products to upload")
        return
    
    client = MongoClient(MONGODB_URI)
    db = client[MONGODB_DB]
    collection = db[MONGODB_COLLECTION]
    
    # Create a unique index on producthunt_url to avoid duplicates
    collection.create_index("producthunt_url", unique=True)
    
    # Insert products with upsert to update existing entries
    inserted_count = 0
    updated_count = 0
    
    for product in products:
        try:
            result = collection.update_one(
                {"producthunt_url": product["producthunt_url"]},
                {"$set": product},
                upsert=True
            )
            
            if result.upserted_id:
                inserted_count += 1
            elif result.modified_count > 0:
                updated_count += 1
                
        except Exception as e:
            logger.error(f"Error uploading product {product.get('title')}: {e}")
    
    logger.info(f"MongoDB upload: {inserted_count} inserted, {updated_count} updated")
    client.close()

def cleanup(extract_dir):
    """Clean up temporary files"""
    import shutil
    try:
        shutil.rmtree(os.path.dirname(extract_dir))
        logger.info(f"Cleaned up temporary directory")
    except Exception as e:
        logger.error(f"Error cleaning up: {e}")

def main():
    try:
        # Get the latest artifact URL
        download_url, artifact_name = get_latest_artifact_url()
        
        # Download and extract the artifact
        extract_dir = download_and_extract_artifact(download_url, artifact_name)
        
        # Process the JSON files
        products = process_json_files(extract_dir)
        
        # Upload to MongoDB
        upload_to_mongodb(products)
        
        # Clean up
        cleanup(extract_dir)
        
        logger.info("Successfully completed data transfer")
        return 0
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.exception("Exception details:")
        return 1

if __name__ == "__main__":
    sys.exit(main())
