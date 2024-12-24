# Word Similarity and Semantic Relationship Analysis (work in progress)

This repository provides a tool designed to analyze the semantic relationships of words. By accepting a single word as input, it returns similar words along with nuanced distinctions such as formality, gender, or plurality. The objective is to go beyond simple word retrieval by offering detailed contextual differences.

### Objective example:

**Input**: `actor`  
**Output**:
- `actress` (female) 
- `performer` (more formal)


### Current implementation:

**Input**: `actor`  
**Output**:
- `actress` (gender) 


The tool uses pre-trained embeddings to calculate semantic similarity and analyze relationships across various predefined dimensions.

## Pre-trained Models
You can utilize any model of your choice. Below are some recommended pre-trained models available for free download. Place the downloaded model in the models/ folder and update the model_path variable in the script accordingly.

Word2Vec Model: [Download Word2Vec Model](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)


GloVe Model: [Download GloVe Model](https://github.com/stanfordnlp/GloVe)


## Jupyter Notebooks

This repository includes Jupyter Notebooks to facilitate exploration and experimentation with the tool’s capabilities.

TODO : Explain a bit of each one


## Directory Structure

Here’s a breakdown of the repository structure
```
.
├── data/                        # Folder for data (future use)
├── experiments/                 # Folder for Jupyter Notebooks
├── models/                            
├── main.py                          
├── README.md                        
├── requirements.txt
├── semantic_dimensions_config.json    
├── semantic_dimensions.py
├── utils.py  
└── visualization_utils.py  
```


## Dependencies

Install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

## Running the `main.py` Script

To execute the main.py script, follow these steps:

1. **Install Dependencies:**
   First, make sure all the required dependencies are installed. You can install them by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Download Pre-trained Model:**
    Download your chosen model (e.g., Word2Vec or GloVe) and place it in the models/ folder. Update the model_path variable in the script to point to the correct location.
   
3. **Run the Script:**
    Execute the script from your terminal using the following command:

    ```
    python main.py <word>
    ```
    Replace `<word>` with the input word for which you want to find similar words and analyze relationships.

## Future Enhancements (TODO)

The ultimate goal of this project is to find the best approach for distinguishing words based on pre-defined, understandable nuances. To achieve this, the following aspects will be explored:

### Key Considerations
1. **Model Selection**

    Currently using Word2Vec and GloVe.
    Future plans include implementing more advanced models such as BERT to capture contextual and nuanced word relationships more effectively.

2. 