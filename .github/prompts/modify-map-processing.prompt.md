---
agent: agent
model: Claude Sonnet 4.5 (copilot)
description: Modify the existing map processing code to include additional data validation, logging, and population assignment features as specified.
tools: ['vscode', 'execute', 'read', 'edit', 'search', 'web', 'agent', 'pylance-mcp-server/*', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/installPythonPackage', 'ms-python.python/configurePythonEnvironment', 'ms-toolsai.jupyter/configureNotebook', 'ms-toolsai.jupyter/listNotebookPackages', 'ms-toolsai.jupyter/installNotebookPackages', 'todo']
---

The objective is to expand the existing map processing script, "process_map_svg.py", primarily by incorporating additional information from the gazetteer csv files without altering the current functionality. As before, the script should match settlements extracted from the .svg with entries in a CSV file, now instead found at /input/gazetteers/empire.csv and /input/gazetteers/westerland.csv for Empire and Wasteland settlements respectively.

Create a single report document to log issues and warnings, including any settlement names found in the .csv that do not have a corresponding settlement in the .svg. It is acceptable for settlements to exist in the .svg that are not in the .csv, but these should be assigned a random population between 100 and 800 using a log-normal distribution as before.

 The .csv files contain 17 columns with one header row, which are in order:
- Settlement: name of the settlement
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

The script should populate geojson files with the standard geojson header (e.g. "type":"Feature", "geometry": {"type":"Point", "coordinates": [<x>, <y>]}) for each settlement in the svg, similar to what the script currently does, where the x, y coordinates are the geospatial lat/long coordinates converted from the inkscape coordinates using the existing formulae. Properties fields should be as follows:
- name: the name of the settlement extracted from the svg text content (tspan) [already implemented in current script]
- province: the province layer the settlement was found in. This should be compared to the name of the province in the csv "Province_2515" column to ensure they match. If they do not match, log a warning. Continue with the province name from the svg layer. [already implemented in current script]
- population: the population of the settlement, extracted from the .csv. If not found, assign a random population between 100 and 800 using a log-normal distribution. [already implemented in current script]
- tags: an array of tags extracted from the "Tags" column in the .csv, splitting on the semi-colon delimiter. If there are any entires in the "Trade" column, add there here with the prefix, "trade:(trade good)". E.g. """textiles; timber""" should yield tags of "trade:timber" and "trade:textiles". If no tags are found, set to an empty array. Tags should always be in the format <tag_type>:<tag_value>, e.g. "source:2eSH", "trade:timber". All tags should be consistent and match a value found in a corresponding array. E.g. for "source", the word after the colon should match a value from ["AndyLaw","2eSH","4eAotE1","4eEiS","4ePBtTC","4eSCoSaS","4eCRB","4eDotRC"]. Implement a test that checks for this format and validity to the array and log a warning if it fails. [modification of existing feature]
- notes: an array of notes extracted from the "Notes" column in the .csv, splitting on the semi-colon delimiter. If no notes are found, set to an empty array. [existing feature]
- size_category: an integer size category assigned based on the population using the following scale:
  - Village (1): 1-300
  - Small Town (2): 301-900
  - Town (3): 901-3000
  - Large Town (4): 3001-15000
  - City (5): 15001-49999
  - Metropolis (6): 50000 and above [modification of existing feature]
- inkscape_coordinates: an array of the original inkscape SVG coordinates [already implemented in current script]
- a "wiki" property group containing four properties:
  - title: from the "wiki_title" column
  - url: from the "wiki_url" column
  - description: from the "wiki_description" column
  - image: from the "wiki_image" column
  If any of these fields are missing in the .csv, set them to null in the geojson. [new feature]


Example output geojson feature for a settlement:
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [
          1.5938634782431125,
          47.177851112667824
        ]
      },
      "properties": {
        "name": "Wissenburg",
        "province": "Wissenland",
        "population": 37008,
        "tags": [
          "source:2eSH"
        ],
        "notes": [],
        "size_category": 5,
        "inkscape_coordinates": [
          519.59229,
          653.02167
        ]
        "wiki": {
          "title": "Wissenburg",
          "url": "https://example.com/wiki/Wissenburg",
          "description": "Wissenburg is a prominent city in Wissenland...",
          "image": "https://example.com/images/wissenburg.jpg"
        }
      }
    },
    ...
  ]
}

Continue all other functionality of the existing process_map_svg.py script as is, including logging invalid text objects found, reporting duplicate settlement names within the same province, and providing a summary report of all processed settlements at the end. Establish some basic tests to verify the new features work as intended, and that the old functionality continues to work. The tests should pass and yield identical results to the previous version of the script where no changes were made. Modify the existing "process_map_svg.py" script to implement these changes. The existing output files and logs can be overwritten - they are backed up externally.