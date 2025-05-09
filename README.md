# MOGAD: Integrated Multi-Omics and Graph Attention for Alzheimer’s Disease’s Biomarker Discovery 
the Multi-Omics Graph Attention Biomarker Discovery Network (MOGAD) aims to efficiently classify diseases and discover biomarkers by integrating various omics data such as DNA methylation, mRNA expression, and miRNA expression.  

![The workflow of our study](workflow.tiff)  
The workflow illustrates the complete workflow of this study, which comprises three parts:  
1) Data collection: We collected three types of omics data and clinical data. Redundant features and noise were removed from the omics data through feature selection;   
2) MOGAD: The omics data and their corresponding cosine similarity matrices are pro-cessed by MGAT and MGAF, respectively. The outputs of these two modules are then fed into AF to produce the final predictions. When clinical data is involved, it is concatenated with the omics data prior to downstream processing;   
3) Biomarker discovery: For each omics feature, its importance was quantified by the decline in prediction performance (F1 score) when the feature was masked in test data. Features were ranked by their impact scores, with higher scores indicating stronger biomarker potential.
  
# Files
**feat_importance.py**： Feature importance functions  
**main_biomarker.py**：Examples for identifying biomarkers   
**main_mogad.py**：Examples of MOGAD for classification tasks  
**models.py**：MOGAD model  
**train_test.py**： Training and testing functions  
**utils.py**：Supporting functions   

# Data
 ROSMAP dataset can be found in https://adknowledgeportal.synapse.org,include DNA methylation(syn3168763),mRNA(syn3505720),miRNA(syn3387327) and clinical data(syn3191087).  
 BRCA dataset can be found in https://xenabrowser.net/datapages/?cohort-GDC%20TCGA%20Breast%20Cancer%20BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443.  
 Hi-C data can be found in http://menglab.pub/hic/.
 
# Usage
We use torch==1.10, cuda==11.3 and python>=3.6 to run this code.  
Use `python /code/main_mogad.py` to disease prediction.  
Use `python /code/main_biomarker.py` to biomarker discovery.  

  
The input data needs to be processed into the following shape：  
1) Numeric values only,
2) Matrix shape should be (samples * features),
3) The row names (samples) of each omics data must correspond to each row of the labels.
