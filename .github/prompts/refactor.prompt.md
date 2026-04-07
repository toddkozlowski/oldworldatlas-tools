---
agent: agent
model: Claude Sonnet 4.6 (copilot)
description: Update and refactor the existing map processing code to standardize the code structure, reduce redunant and exceptional code, and prepare for further features.
tools: ['vscode', 'execute', 'read', 'edit', 'search', 'web', 'agent', 'pylance-mcp-server/*', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages', 'todo']
---

Update and refactor the existing map processing code "scripts/process_map_svg.py" to standardize the code structure, reduce redunant and exceptional code, and prepare for further features. This should be done in a way that does not alter the current functionality of the script, but rather optimizes and standardizes the codebase to make it easier to maintain and expand in the future. Develop and interactive CLI when the script is executed that prompts/guides the user through optional stages or steps of the processing, such as downloading and validating the gazetteer .csv files, processing the .svg file, and generating the geojson output. This should be designed in a way that allows for easy addition of new steps or features in the future without requiring significant restructuring of the codebase. This should also allow for folding the various one-off scripts into the main codebase as optional steps in the workflow.

When in doubt, look to the already existing and functioning code in "scripts/process_map_svg.py" for guidance on how to implement the refactored workflow, and how to handle any potential edge cases or issues that may arise during the processing. The goal of this refactor is not to change the core functionality of the script, but rather to optimize and standardize the code structure, reduce redundancy, fold in one-off scripts, develop a user-friendly CLI, and prepare for future features and expansions. Therefore, it is important to ensure that all existing functionality is preserved and that any changes made do not introduce new bugs or issues into the processing workflow.

The refactored workflow should look like:
# Parsing and processing workflow for map data
1. download and update gazetteer .csv files from the google sheets source (one link per gazetteer), and validate the format of the .csv files to ensure they contain the expected columns and data types. Log and error flag any issues with the .csv files, such as missing columns or invalid data formats. Store the last three versions of the .csv files in a version history for reference and potential rollback, with the filename appended with the date-time of the download (e.g. "empire_2024-06-01T12-00-00.csv"). After the third version, delete the oldest backup version when a new one is downloaded and confirmed to be without errors. This is similar to the functionality of the existing "scripts/sync_gazetteers.py" script, but should be integrated into the main workflow of the script rather than being a separate one-off script.

## List of expected human settlement gazetters:
- empire.csv
- westerland.csv
- bretonnia.csv
- tilea.csv
- estalia.csv
- norsca.csv
- border_princes.csv
- kislev.csv

Each of these gazetters should have the following columns:
- Settlement: name of the settlement [must have a value for each entry, must be a string]
- Population: population of the settlement
- Estate: estate the settlement belongs to
- Trade: main trade good of the settlement
- Tags: tags associated with the settlement. These are in a semi-colon delimited format, e.g. """source:2eSH; source:4ePBtTC"""
- Notes: any additional notes about the settlement
- Coordinates: geographic coordinates of the settlement in "longitude latitude" format
- Province_2515: province to which the settlement belongs in 2515
- Province_2512: province to which the settlement belongs in 2512
- Province_2276: province to which the settlement belongs in 2276
- Ruler_2515: ruler of the settlement in 2515
- Ruler_2512: ruler of the settlement in 2512
- Ruler_2276: ruler of the settlement in 2276
- wiki_url: URL to the settlement's wiki page
- wiki_title: title of the settlement's wiki page
- wiki_description: description of the settlement from the wiki
- wiki_image: URL to an image of the settlement from the wiki

## List of nonhuman settlement gazetteers:
- karaz_ankor.csv
- wood_elves.csv
- skavendom.csv
These gazetteers have slightly exceptional columns, which should be handled with the same validation and logging as the standard columns, but with the understanding that these columns may not be present in all gazetteers:
- karaz_ankor.csv and wood_elves.csv are formatted the same as the standard gazetteers but have an additional 'Type' column to the left of 'Population'
- 'skavendom.csv' has columns 'Settlement', 'Type', 'Population', 'Major Clan(s)', 'Minor Clan(s)', 'Trade', 'Tags', 'Notes', 'Coordinates', 'wiki_url', 'wiki_title', 'wiki_description', 'wiki_image' (i.e. it does not have the province or ruler columns)

## List of non-settlement gazetteers:
- provinces.csv
-- with columns: name [required, string], formal_title, part_of, population, province_type [required, must be one of "Nation", "Major Division", "Minor Division"], local_category, wiki_url, wiki_title, wiki_description, wiki_image
- geographic_feature_labels.csv
-- with columns: name [required, string], type [required, must be one of "Ocean", "Major Sea", "Large Sea", "Medium Sea", "Small Sea", "Lake", "Large Wetland", "Small Wetland", "Large River", "Medium River", "Small River", "Large Forest", "Small Forest", "Other"], wiki_url, wiki_title, wiki_description, wiki_image
- points_of_interest.csv
-- with columns: name [required, string], type [required, must be one of "City District", "Taverns and Inns", "Forts and Castles", "Monasteries and Temples", "Chaos Shrine", "Other"], description, tags, wiki_url, wiki_title, wiki_description, wiki_image

2. extract settlement and POI data from the .svg file, with the final goal of importing all relevant data into a series of structured .geojson files. The .geojson files should have the following structure (dummy data):

for settlements:
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [
          -32.8399321,
          45.14070635,
        ]
      },
      "properties": {
        "name": "Zweikalten",
        "province": "Wissenland",
        "population": 1089,
        "tags": [
          "source:2eSH",
          "source:4ePBtTC"
        ],
        "notes": [],
        "size_category": 3,
        "wiki": {
          "title": null,
          "url": null,
          "description": null,
          "image": null
        }
      }
    },
    ...

for points of interest:
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [
          -1.1958525550889592,
          50.590772347227194
        ]
      },
      "properties": {
        "name": "The Old Bögenauer",
        "type": "Taverns and Inns",
        "tags": [
          "source:2eSH",
          "source:4ePBtTC"
        ],
        "wiki": {
          "title": null,
          "url": null,
          "description": null,
          "image": null
        }
      }
    },
    ...

for province labels:
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [
          3.1508593896032004,
          52.884770181392
        ]
      },
      "properties": {
        "name": "TALABHEIM",
        "province_type": "Province",
        "formal_title": "Imperial County and City-State of Talabheim",
        "part_of": "Talabecland",
        "population": 0,
        "wiki": {
          "title": "Talabheim",
          "url": "https://warhammerfantasy.fandom.com/wiki/Talabheim",
          "image": null,
          "description": "Talabheim, known in ancient times as Taalaheim..."
        },

      }
    },

for geographic feature labels:
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [
          -1.1958525550889592,
          50.590772347227194
        ]
      },
      "properties": {
        "name": "The Old Bögenauer",
        "type": "Taverns and Inns",
        "description": "...",
        "wiki": {
          "title": null,
          "url": null,
          "description": null,
          "image": null
        }
      }
    },
    ...

Settlements, POI, province labels, and geographic feature labels are all represented as text elements (tspan) in the .svg file, placed at specific coordinates corresponding to their location in geospatial units of latitude and longitude. The scale of the Inkscape canvas is such that 1 user unit corresponds to 1 degree latitude or longitude (map is in equirectangular projection). Therefore, the coordinates of the text elements can be directly extracted and used as the geographic coordinates for the corresponding settlement, POI, province label, or geographic feature label, provided that any transforms applied to the elements or their parent groups/layers are properly accounted for / compensated for in the coordinate extraction process.

Settlement text elements are found in the .svg file with the following layer structure:
- layer: "settlements" > layer: "<nation name>" where nation name is one of "empire", "westerland", "bretonnia", "tilea", "estalia", "norsca", "border_princes", "karaz_ankor", "wood_elves", "skavendom", or "kislev" > (optional, intermediate layers) > text element

There may be additional layers between "<nation name>" and the text elements. If there is, the name of the layer directly below <nation name> should be extracted and stored for the "province" property of the settlement in the geojson output. If there are two or more intermediate layers, the name of the layer directly above the text element should be extracted and stored for the "estate" property of the settlement in the geojson output. If there are no intermediate layers, the province and estate properties for that settlement should be left blank in the geojson output (unless populated from the gazetteer .csv files in the later stages of the workflow).

Points of interest text elements are found in the .svg file with the following layer structure:
- layer: "points_of_interest" > layer: "<type>" where type is one of "City District", "Taverns and Inns", "Forts and Castles", "Monasteries and Temples", "Chaos Shrine", or "Other" > text element
The name of the <type> layer should be extracted and stored for the "type" property of the point of interest in the geojson output.

The province label text elements are found in the .svg file with the following layer structure:
- layer: "province_labels" > "<province size category>" > text element, where province size category is one of "major_nation", "minor_nation", "major_division", "minor_division", or "other"
The name of the <province size category> layer should be extracted and stored for the "province_type" property of the province label in the geojson output.

Make sure to handle any potential issues with the .svg file, such as missing or malformed elements, and log any errors or warnings encountered during the extraction process. Furthermore, check any and all layers for layer, group, or element transforms which may affect the coordinates of the settlements, and apply the necessary transformations to ensure the coordinates are accurate. Check how the current script handles this. 

3. Merge the data extracted from the .svg file with the data from the gazetteer .csv files, using the settlement / point of interest / province label / geographic feature label name as the primary key for merging. For settlements, there may be more than one settlement with the same name, in which case settlement name and province should be used as the primary key for merging. Additionally log if any province contains more than one settlement with the same name. This should be done in a way that allows for easy handling of any discrepancies or mismatches between the two data sources. It is expected that all items in the csv files should have a corresponding text element in the .svg file, but it is not required that all text elements in the .svg file have a corresponding entry in the .csv files. Log any elements in the csv files that do not have a corresponding text element in the .svg file. The csv files should be considered the primary source of truth for the settlement / point of interest / province label / geographic feature label data, and any discrepancies should be resolved in favor of the data from the csv files.

The csv gazetteers should be used to provide the additional properties for the settlements, points of interest, province labels, and geographic feature labels in the geojson output, such as population, tags, notes, wiki information, etc. The data from the .svg file should primarily be used to provide the geographic coordinates for the corresponding items in the geojson output.

4. Generate the final geojson output files for settlements, points of interest, province labels, and geographic feature labels, with the standardized structure as described above. Make sure to handle any potential issues during the generation process, such as missing or malformed data, and log any errors or warnings encountered. The generated geojson files should be saved in a specified output directory, with clear and consistent naming conventions (e.g. "settlements_empire.geojson", "points_of_interest.geojson", "province_labels.geojson", "geographic_feature_labels.geojson"), with one geojson file for each csv gazetteer. There should be an additional prompt to the user at this stage, asking if they would like to "push" the generated geojson files to a specified directory, and if so, the files should be copied to that directory (e.g. a directory in a github repository for the map data). Doing so should first move the existing files in the destination directory to a backup directory, e.g. ("/backup") with the current date-time appended to the filename (e.g. "settlements_empire_backup_2024-06-01T12-00-00.geojson"), and then copy the newly generated geojson files to the destination directory. After copying, log a summary of the changes, including any new files added, any files that were overwritten, and any files that were moved to the backup directory. Store the last three versions of the geojson files in a version history for reference and potential rollback. After the third version, delete the oldest backup version when a new one is generated. This behaviour is similar to the code in the file 'scripts/deploy_gazetteers.py', but should be integrated into the main workflow of the script rather than being a separate one-off script.

5. The population category for the settlement geojsons should be extracted from the "Population" column in the gazetteer csv files, which contains the population of the settlement as an integer. This should be converted into a population category on a scale of 1 to 6, with the following thresholds:
        if population <= 300:
            return 1  # Village
        elif population <= 900:
            return 2  # Small Town
        elif population <= 3000:
            return 3  # Town
        elif population <= 15000:
            return 4  # Large Town
        elif population <= 49999:
            return 5  # City
        else:
            return 6  # Metropolis

If there is no population data for a settlement, or there is no corresponding entry for that settlement in the gazetteer csv files, the population should be randomly assigned. Adopt the method currently in use by the process_map_svg.py script. 

# Additional features:
1. Fold the existing "label_settlements.py" script into the main codebase, which serves to rename the inkscape text element label to be the same as the contents of that text element.

2. Incorporate elements of 'extract descriptions.py' and 'download_wiki_metadata.py' scripts as optional steps in the main workflow. This should check each and every entry in all the gazetteer csv files if there exists a Warhammer wiki article with a title that matches the name of the settlement / point of interest / province label / geographic feature label, and if so, extract the description and image from the wiki article and include it in the corresponding entry in the geojson output. Skip checks on any entries that already have wiki information in the csv files. At the end of this step, ask the user if the csv files should be updated with the newly extracted wiki information, and if so, update the csv files accordingly. This should be the only time the csv files are modified in the workflow.

# Report Log
should contain information on:
- any issues with the gazetteer csv files during the validation process, such as missing columns or invalid data formats
- any discrepancies or mismatches between the data in the gazetteer csv files and the text, e.g. elements in the csv files that do not have a corresponding text element in the .svg file
- any repeated settlement names within the same province, along with the province name and the number of occurrences of that settlement name within that province
- summary of the number of settlements of each nation (and province if possible), points of interest, province labels, and geographic feature labels processed and included in the final geojson output
