# scraper

4/4/25 
Updated scraper.py and product_hunt_scraper.yml 
- Improved Artifact Management: Each workflow run now creates a timestamped directory for its output
- Isolated Data Collection: JSON files from each run are stored in separate directories (e.g., product_hunt_metadata/run_20250404_123456)
- No More Duplicate Downloads: Artifacts now contain only new data from the current run
- Run Index System: Added a runs_index.json file that tracks metadata for all workflow runs
- Better Organization: Each run's data is cleanly separated, making it easier to identify when products were scraped
- Reduced Artifact Sizes: Downloads are now smaller and focused only on newly scraped data
- Improved Logging: Run summaries now include information about the specific run directory
- Database Consistency: The deduplication database continues to prevent re-scraping previously processed URLs
