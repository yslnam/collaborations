# Mofo Bikes


## Directory structure
```
├── README.md           <- Top-level README for project members
│
├── data                <- Layers of data pipeline, excluding models (stored separately)
│   ├── 00_raw          <- Immutable raw data
│   ├── 01_intermediate <- Cleaned version of raw
│   ├── 02_processed    <- Data used for modelling
│
├── notebooks           <- Jupyter notebooks
│
├── models              <- Trained models and model output stored for local use
│                                           
├── .gitignore          <- Avoids uploading data, credentials, outputs, system files etc
│
└── src                 <- Source code for use in this project.
    ├── __init__.py     <- Makes src a Python module
    │
    ├── d00_utils       <- Functions used across the project
    │   └── remove_accents.py
    │
    ├── d01_preparation <- Scripts to a) retrieve, read, and write data, and
    │                      b) transform data from raw to intermediate, etc.
    │   └── load_data.py
    │   └── create_intermediate_data.py
    │
    ├── d02_processing <- Scripts to turn intermediate data into modelling input
    │   └── create_master_df.py
    │
    ├── d03_modeling  <- Scripts to a) train models and then use trained models to make
    │                    predictions, and b) analyse model performance and model selection
    │   └── train_model.py
    │   └── calculate_performance_metrics.py
    │    
    ├── d04_reporting  <- Scripts to a) produce reporting tables, and b) create plots
    │   └── create_summary_tables.py
    │   └── create_data_visualizations.py
```
## Project workflow

1. **Extract, Transform, and Load (ETL)**: extract data from its source, then transform and load it.
2. **Exploratory data analysis (EDA)**: explore the data.
2. **Pre-process data**: Impute missing values, select training and testing sets, etc.
4. **Create features**: Recombine and enrich the data to create features aiding the modeling.
4. **Train the model(s)** 
5. **Assess performance on the test set** using an appropriate metric, and then examine the performance of your model "out of sample."
6. **Analyze output and create data visualizations**: create summary tables and data visualizations for analysis and final reporting.

### Notes
#### Source code workflow
1. Prototype code in a jupyter notebook
2. Move code into a function that takes data and parameters as inputs and returns the processed data or trained model as output
3. Test the function in the jupyter notebook
4. Move the function into the src folder
5. Import the function in the jupyter notebook 
6. Test the function is working

Functions can be imported into a notebook as follows.

1. Tell the notebook where the functions are:

    ```
    import os
    import sys
    src_dir = os.path.join(os.getcwd(), '..', 'src')
    sys.path.append(src_dir)
    ```

2. State which functions to import

    ```
    from d00_utils.my_fun import my_fun
    ```
    
#### Code pipeline
The code that produces the different layers of the data pipeline should be abstracted into functions.

A *code pipeline* is a set of code that handles all the computational
tasks your project needs from beginning to end. The simplest
pipeline is a set of functions strung together. Example:

    int_data = create_int_data(raw_data)
    pro_drug_features = create_pro_drug_features(int_data)
    pro_patient_features = create_pro_patient_features(int_data)
    pro_master_df = create_pro_master_df(pro_drug_features, pro_patient_features)
    
Each step is broken down into a number of subsets creating pipelines for each layer of the data pipeline.
The end-to-end pipeline is then the concatenation of the sub-pipelines. 

## Credits
> Adapted from [*The Hitchhiker's Guide to Data Science for Social Good*](https://github.com/dssg/hitchhikers-guide)
