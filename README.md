# Word Similarity and Semantic Relationship Analysis (in progress)

This repository provides a tool designed to receive a single word as input and return its synonyms, along with any nuanced distinctions between them. The objective is to not only find synonyms but also to provide additional information on subtle differences in meaning, such as formality, gender or plurality.

### Objective example:

**Input**: `actor`  
**Output**:
- `actress` (female) 
- `performer` (more formal)


### Last implementation:

**Input**: `actor`  
**Output**:
- `actress` (female) 
- `performer` (more formal)


The tool leverages pre-trained **Word2Vec embeddings** to compute the semantic similarity between words and analyze their relationships across various predefined dimensions.

### Pre-trained Model

A pre-trained Word2Vec model is required to use this tool. You can download the model from the following link:

[Download Word2Vec Model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)


### Notebooks

The **Jupyter Notebooks** in this repository are key to experimenting with the tool’s capabilities. 

In particular, the `semantic_dimensions.ipynb` notebook is crucial for exploring how the model analyzes word relationships along different semantic axes like formality and gender.


### Directory Structure

Here’s a breakdown of the repository structure
```
.
├── data/                        # Folder for data (will be used in the future)
├── experiments/                 # Folder for Jupyter Notebooks
├── models/                      # Here is where the models are        
├── main.py                          
├── README.md                        
├── requirements.txt                 
└── utils.py   
```


### Dependencies

To run this project, you'll need to install the required dependencies. You can do this by running the following command:

```
pip install -r requirements.txt
```

## Running the `main.py` Script

To run the `main.py` script, follow these steps:

1. **Install Dependencies:**
   First, make sure all the required dependencies are installed. You can install them by running:

   ```bash
   pip install -r requirements.txt
   ```
   
2. **Running the Script:**
    Once the dependencies are installed and the model is downloaded, you can run the script from your terminal. Use the following command:

    ```
    python main.py <word> --N <number>
    ```
    - Replace <word> with the word you want to find synonyms for.
    - Replace <number> with the number of top synonyms you want to retrieve (this is optional, defaults to 5).

3. TO DO