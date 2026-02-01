---
agent: agent
model: Claude Sonnet 4.5 (copilot)
description: Modify the existing map processing code to include additional data validation, logging, and population assignment features as specified.
tools: ['vscode', 'execute', 'read', 'edit', 'search', 'web', 'agent', 'pylance-mcp-server/*', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages', 'todo']
---

The objective is to write a script which queries the Warhammer Fandom Wiki (https://warhammerfantasy.fandom.com/) for each settlement, region, or point of interest that has a wiki article. Populate a file with the URL, and the Open Graph title, description and image url.

First, determine whether or not the given settlement has a wiki page, e.g. "https://warhammerfantasy.fandom.com/wiki/Nuln". If it does, extract the Open Graph metadata from the page HTML, specifically the following meta tags:
- og:title
- og:description
- og:image
If these are present, store them in the 'empire.csv' in the columns appropriately. If the page does not exist, leave these fields blank.

Ensure that the script handles HTTP errors gracefully, logging any issues encountered during the requests to a separate log file for later review.
The final output should be a csv file with the four fields (url, title, description, image url). Once the entire script is finished running and that .csv is populated, prompt the user to update the 'empire.csv' file with the new metadata columns populated where applicable, in the columns titled "wiki_url", "wiki_title", "wiki_description", and "wiki_image". Make sure no other fields in the CSV are disturbed. 

Be respectful of the wiki's usage policies, implementing appropriate rate limiting to avoid overwhelming their servers with requests. If deemed appropriate, slow down the request rate to one request every two seconds. Generate a summary report at the end of the script's execution, detailing how many settlements were processed, how many wiki pages were found, and any errors that occurred. Additionally, provide a live log to the terminal of progress as the script runs, indicating which settlement is currently being processed (also settlement number out of total), the number of successful fetches, and any errors encountered in real-time.

If the settlement name contains a non-latin character (e.g., "Bögenhafen"), try an additional request for the name with the non-latin character replaced by its closest latin equivalent (e.g., "Bogenhafen") before concluding that no wiki page exists for that settlement.