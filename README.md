# Data Science Project - 2025

# Setup
1. Download the [Nebraska 2017 **All Records** zip File](https://files.consumerfinance.gov/hmda-historic-loan-data/hmda_2017_ne_all-records_labels.zip)
2. Extract the `.zip` file
3. Rename the extracted `.csv` file to `hmda_ne_all_2017.csv`
4. Place the `.csv` file into `/data`
5. `cd` into the cloned folder of the GitHub repository
6. Run `python -m venv project`
7. Run `project/Scripts/activate`
8. Run `pip install -r requirements.txt`

# EDA Script:
1. Run `cd scripts`
2. Run `python hmda_eda.py --input ../data/hmda_ne_all_2017.csv --outdir ../reports/hmda_eda`
3. All the figures and the reports will be generated in `../reports/hmda_eda` folder.

# Selection of relevant features via the Analyser
The `naive_missing_analyser.py` loads a CSV file, calculates missing value percentages per column,  identifies column data types (numeric or categorical),  prints the results, and optionally saves them to a CSV file.
Command to run:
`python naive_missing_analyser.py ../data/hmda_ne_all_2017.csv --save occupancy.csv`

## Insights gained by the Analyser:

### Pruning features which have a high missing percentage:
We pruned features `1-25` as `> 97%` of the rows were missing values for those features (The threshold is highlighted in gray).
![PruneThresholdForMiss](readme_media/PruneThresholdForMiss.png)


### Pruning features which are just identifiers
The features highlighted in orange do not add any relevant information for exploratory data analysis (they would be useful for inference and hypothesis testing as they just are Numeric forms of their String counterparts)
![PruningIdentifiers](readme_media/PruningIdentifiers.png)

### Remaining Dataset
![RemainingDataset](readme_media/RemainingDataset.png)
