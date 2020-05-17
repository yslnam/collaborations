# Data Pipelines

Structuring data into multiple layers via directory structure:

- **Raw**: Raw data only. To quote from workflow package [Data Science Cookiecutter](https://drivendata.github.io/cookiecutter-data-science/#data-is-immutable):

> Don't ever edit your raw data, especially not manually, and especially not in Excel. 
> Don't overwrite your raw data. 
> Don't save multiple versions of the raw data. 
> Treat the data (and its format) as immutable. 
> The code you write should move the raw data through a pipeline to your final analysis. 
> You shouldn't have to run all of the steps every time you want to make a new figure (see Analysis is a DAG), but anyone should be able to reproduce the final products with only the code in src and the data in data/raw.

- **Intermediate**: Cleaned version of raw data. Storage formats: `postgres` or `parquet`

- **Processed**: Includes input data with combined and enriched features, as well as processed data to train models. Storage formats: `postgres`/`parquet` (input data) and `pickle` (processed data)

## Note
Models are stored elsewhere