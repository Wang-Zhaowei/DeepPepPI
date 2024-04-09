## DeepPepPI: a deep cross-dependent framework with information sharing mechanism for predicting plant peptide-protein interactions

This repository contains the source code and benchmark datasets used in this paper.

## Introduction

Motivation:&#x20;

Peptide-protein interactions (PepPIs) play a crucial role in various fundamental biological activities in plants. Due to the disadvantages of biological experimental technologies such as time-consuming and labor-intensive, it is indispensable to develop computational methods for identification of PepPIs. However, less attention has been paid to represent the pair-wise interaction information between peptides and proteins in existing methods, limiting the prediction performance.

Results:

In this paper, we present DeepPepPI, a novel deep cross-dependent framework for accurate prediction of plant PepPIs. Concretely, a data-driven context-embedded representation (DCR) module is developed as the peptide feature extractor that can capture rich contextual semantic information, even from short sequences. To fully characterize inherent properties of proteins, we propose a bi-level self-correlation search (BSS) module, which integrates the primary sequence and secondary structure into a unified space to learn their potential relationships. In addition, a cross-dependent feature integration (CFI) module with information sharing mechanism is introduced, aimed at providing a comprehensive feature representation to portray the intricate interaction patterns between peptides and proteins from a global perspective. Comprehensive experiments demonstrate that the proposed method achieved excellent prediction performance and powerful generalization capability, which can serve as an effective and convenient tool for the characterization and identification of PepPIs.

## DeepPepPI

![DeepPepPI](./Overflow.png)

## Dataset

In this paper, the PepPI datasets are extracted from the widely used STRING database, in which the sequences with no more than 100 amino acid residues are identified as peptides. For positive samples, the interactions between peptides and proteins were filtered by setting a cut-off value greater than 950 for the “*combined\_score*” to reduce the number of false positives. Besides that, we constructed the negative set by randomly shuffling the non-interacting pairs of peptides and proteins, meaning that there were no records of their “*combined\_score*” in STRING database. For training and evaluating the prediction performance of DeepPepPI, we randomly picked up 80% of the samples as the training datset and the remaining samples were taken as the independent testing dataset to ensure that there was no systematic difference between training and independent testing samples.

*   Species: *Arabidopsis thaliana* (BD1) and *Solanum lycopersicum (BD2)*.

##### Peptide embedding features:&#x20;

We introduce ProtT5-XL-UniRef50 (ProtT5) to capture the contextual information of peptide sequences, which can be available at <https://github.com/agemagician/ProtTrans>.

*   Following the steps listing in the **ProtT5-XL-UniRef50.ipynb**.

*   The obtrained embeddings are saved as .**npy**.

##### Protein secondary structure:

The SOPMA web server (<https://npsa-prabi.ibcp.fr/cgi-bin/npsa_automat.pl?page=npsa%20_sopma.html>) is used to predict protein secondary structures consisting of three classical conformational states: α-helix (H), β-sheet (E) and coil (C)..

### Setup and dependencies

*   Python 3.7

*   Keras >\= 2.10.0

*   TensorFlow >\= 2.10.0

*   Numpy

*   Scikit-Learn

### Code details

*   DeepPepPI.py: train and evaluate the model.

*   Model.py: DeepPepPI modules.

*   multi\_head\_att.py: multi-head self-attention mechanism.

*   load\_data.py: data importing and processing.

*   test\_scores.py: performance calculation.

```python
python DeepPepPI.py
```

## Citation

Zhaowei Wang, Jun Meng, Qiguo Dai, et al. "DeepPepPI: A deep cross-dependent framework with information sharing mechanism for predicting plant peptide-protein interactions."   ***Expert Systems with Applications*** (2023) \[*Under Review*]
