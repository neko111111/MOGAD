# MOGAD: Integrated Multi-Omics and Graph Attention for Alzheimer’s Disease’s Biomarker Discovery 
the Multi-Omics Graph Attention Biomarker Discovery Network (MOGAD) aims to efficiently classify diseases and discover biomarkers by integrating various omics data such as DNA methylation, mRNA expression, and miRNA expression.  

![The workflow of our study](Figure1-v4.tiff)  
Figure1-v4 illustrates the complete workflow of this study, which comprises three parts:  
1) Data collection: We collected three types of omics data and clinical data. Redundant features and noise were removed from the omics data through feature selection;   
2) MOGAD: The omics data and their corresponding cosine similarity matrices are pro-cessed by MGAT and MGAF, respectively. The outputs of these two modules are then fed into AF to produce the final predictions. When clinical data is involved, it is concatenated with the omics data prior to downstream processing;   
3) Biomarker discovery: For each omics feature, its importance was quantified by the decline in prediction performance (F1 score) when the feature was masked in test data. Features were ranked by their impact scores, with higher scores indicating stronger biomarker potential.
Input shape:(samples,features)
# Files
**feat_importance.py**： Feature importance functions  
**main_biomarker.py**：Examples for identifying biomarkers   
**main_mogad.py**：Examples of MOGAD for classification tasks  
**models.py**：MOGAD model  
**train_test.py**： Training and testing functions  
**utils.py**：Supporting functions   
