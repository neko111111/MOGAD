# MOGAD: Integrated Multi-Omics and Graph Attention for Alzheimer’s Disease’s Biomarker Discovery
Zhizhong Zhang<sup>1, †</sup>, Yuqi Chen<sup>1, †</sup>, Changliang Wang<sup>2, †</sup>, Maoni Guo<sup>2</sup>, Lu Cai<sup>1</sup>, Jian He<sup>1</sup>, Yanchun Liang<sup>3</sup>, Garry Wong<sup>4</sup>, Liang Chen<sup>1, *</sup>  
the Multi-Omics Graph Attention Biomarker Discovery Network (MOGAD) aims to efficiently classify diseases and discover biomarkers by integrating various omics data such as DNA methylation, mRNA expression, and miRNA expression.  

![The workflow of our study](/MOGAD/Fugurev1-4.tiff)  
Figure illustrates the complete workflow of this study, which comprises three parts: 1) Data collection: We collected three types of omics data and clinical data. Redundant features and noise were removed from the omics data through feature selection; 2) MOGAD: The omics data and their corresponding cosine similarity matrices are pro-cessed by MGAT and MGAF, respectively. The outputs of these two modules are then fed into AF to produce the final predictions. When clinical data is involved, it is concatenated with the omics data prior to downstream processing; 3) Biomarker discovery: Refer to Biomarker discovery section.

# Files
**feat_importance.py**： Feature importance functions  
**main_biomarker.py**：Examples for identifying biomarkers   
**main_mogad.py**：Examples of MOGAD for classification tasks  
**models.py**：MOGAD model  
**train_test.py**： Training and testing functions  
**utils.py**：Supporting functions   
