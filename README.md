
# Repo Overview

## Important Files/Directories

* data/cagi5_mpra:
  contains reference sequences for all enhancers (base_seqs.py), plus deepsea predictions for ref and alt alleles for all of the training data.
* data/cagi5_df.csv:
  the training data with added conservation information converted into csv format
* data/release.xlsx:
  raw training data
* conservation/
  a clone of some of the code from the keras codebase used for the deepsea prediction tasks included to make importing models trained on that task easier
* crossval.py
  helpers for cross validation (see below)
* models.py
  helpers for featurisation/prediction (see below)
* utils.py
  generic helper functions (including some relating to data from the previous year's challenge (i.e. cagi4))
* cagi5_utils.py
  cagi5-specific helper-functions 

# Notes on Code

The code basically consists of helpers for pre-processing the data, helpers for cross-validation and helpers for featurisation/prediction.

Examples of the full cross-validation training + evaluation process are in the notebook Cagi5CV.ipynb


## Setup

N.B. this is only necessary if you want to redo any preprocessing or generate features using models I've trained on the deepsea data whose weights are saved in my scratch space.
All the data needed to perform cross validation using DeepSEA features is in this repo already.

To enable the loading of keras model weights and genomic data from my scratch drive, I use two empty subdirectories within this repo: data/remote_results and data/remote_data, which are excluded from git.

Depending on whether I'm working on my laptop or directly on the version of this repo I maintain in my hpc workspace I then use sshfs / symbolic links to point

  * data/remote_results to /scratch/arh96/consresults
  * data/remote_data to /scratch/arh96/conservation/data


## Pre-processing

utils.save_base_seqs saves the reference sequences in data/cagi5_mpra/base_seqs.csv, based on genomic coords

read_xlxs.py converts the excel formatted data to a pandas df.

utils.get_cons_scores
returns conservation scores which must be added to the df for models including conservation features to work. The version of the data frame saved in the repo currently has these included.


## Cross-validation

The idea of the cross validation procedure is to randomly split each enhancer’s contiguous chunks into k folds. We then perform cross validation, by training a model on the data from all enhancers with the ith fold left out, for 1 <= i <= k. The predictions on the held out folds are concatenated to generate a set of predictions for the full set of training data, which is returned in a DataFrame.

cagi5_utils.get_breakpoint_df is what I use to tag the separate contiguous chunks in each enhancer with ‘chunk_ids’. It adds this information plus the length of the chunk as columns to the original df.

crossval.df_cv_split works on df augmented with this chunk information, performing the random allocation of chunks to folds, depending on k (i.e. the number of cv folds specified). It returns a dictionary mapping enhancer names to a length-k list of lists of the chunk ids in each fold.

crossval.ChunkCV is a class which handles the cross validation process, accepting as arguments the training dataframe (including chunk info), and parameters specifying the type of model to be trained.

N.B. to use a pre-defined set of splits (folds) , pass a dictionary mapping enhancers to lists of chunks in each fold (the output of crossval.df_cv_split) to ChunkCV using the fold_dict kwarg. Otherwise ChunkCV will randomly compute a new set of folds each times, making comparisons of cv metrics across multiple instances of ChunkCV using different models or repeated runs of a single ChunkCV meaningless.

crossval.ChunkCV.get_cv_preds() returns the training data frame with an extra column , cv_prediction, corresponding to the predicted class of that datapoint made by the model trained on the folds other than the one including the chunk to which the datapoint belongs. This dataframe can then be used to calculate cv metrics, either by averaging fold scores, or by computing a single score globally (i.e. by passing df['class'], df['cv_prediction'] as y_true, y_pred to some sklearn scoring metric).

crossval.cvpreds_df_enhancer_folds plays a similar role to ChunkCV, but the cross-validation proceeds by completing leaving out each enhancer in turn , rather than just a selection of chunks from each enhancer. This makes less sense as a CV procedure, but was what I implemented first.


## Models

Inheriting from the BaseModel class in models.py, individual model classes implement a single method, get_features, which takes as input a training data frame, and returns a numpy array representing the feature matrix.

the fit method of the parent BaseModel class then uses the SKLearn interface to train either a logistic regression or xgboost classifier using these features. 

I’ve currently written it so that there is a single class per feature type (i.e. there is one class for getting deepsea prediction features from a saved keras model, another class for getting conservation features, etc.). Different sets of features can be combined into a single model by passing the classes implementing the desired featurisations to the MixedModel class.

### Getting DeepSea task predictions as features

There are two classes that can be used here (defined in models.py):

  1. DeepSeaSNP: to use the original DeepSea model (whose predictions for ref and alt alleles for the full training dataset are included in the repo as data/cagi5_mpra/deepsea_ref_preds.npy and data/cagi5_mpra/deepsea_alt_preds.npy)

  2. DSDataKerasModel: to use predictions from a saved keras model trained on the deepsea data.


I’ve found that using the set of 919 difference features (ref_pred - alt_pred) for each deepsea task works better than other alternatives (absolute difference, log odds, difference + log odds). The choice of feature type(s) to use can be specified via the feattypes argument to DSDataKerasModel/DeepSeaSNP, which should be a list containing as entries any subset of: ‘diff’, ‘absdiff’, ‘odds’, ‘absodds’, ‘scaleddiff’. The default (which I expect to generally be the best, especially if we want to predict the direction of the effect) is just [‘diff’]

To generate these features for a given training dataframe, the DSDataKerasModel class will first look for a set of saved predictions in the directory 'data/cagi5_mpra/EXPERIMENT_ref_preds.npy', where EXPERIMENT is the name of the deepsea experiment to be used. If this is not found, it will generate the predictions on the fly. Doing this relies on a couple of helper functions.

  1. ‘ref_sequence’ and ‘alt_sequence’ columns containing the first two outputs of utils.get_check_sequences_cagi5, and representing the full sequences for ref and alt alleles must be added to the data frame before passing it to the CV handler (ChunkCV). get_check_sequences_cagi5 uses the enhancer id to get the reference sequence from the set of reference sequences saved in the first preprocessing step above, and from this sequence computes the appropriate full alternate allele sequence based on the genomic coordinates of the variants and of the start/end points of the enhancer sequence.

  2. DSDataKerasModel.get_features method will generate ref and alt predictions for each variant, then invokes utils.snp_feats_from_preds to convert these into a score for each variant, based on the type of comparison (diff/log odds..) specified via the feat types argument (see above).


## Modelling Ideas

  - Model each enhancer separately
  - Smooth the conservation scores over different length scales
  - Spatially smooth the predictions
  - Use more features from ENCODE (cell types)


## Advice from Elena Vigorito

Thanks for the meeting. I can only give you some biological advice, which may not be useful.

  1. If the size of the promoter and enhancers cloned in the vector is small (dont
     remember) I dont think you will get the chromatin marks that you get
     from endogenous regions, so likely that info wont be very useful, but it
     may be worth looking at the literature.

  2. I think transcription binding sites are useful and I would check which
     ones are present and occupied  from encode chip-seq data. I would also
     check that in the cell type that they are doing the experiments those
     transcription factors are expressed (very common cell lines, RNA-seq
     data).  Also, I would try to look for databases that tells altering
     mutations for transcription factors binding sites.


## Other Advice

Lorenz suggested looking at effect sizes associated with different nucleotide
substitutions. I have added some exploratory analysis that seems to show this
is a real effect and have also added features for this.

Paul Newcombe suggested integrating information from eQTL studies, perhaps
the GTEx one in particular but realised that these will only cover a small
fraction of substitutions.

Maria suggested modelling the spatial effects with some latent variable process
such as a hidden Markov model.


# Related publications

## Tewhey et al. 2016

Found significant association between active sequences (one of two alleles has altered (raised)
expression) and DNase hypersensitive regions, also with chromatin marks and with TF-binding
profiles but this was all cell type specific. emVars refers to sequences with allelic skew. This
showed modest but reproducible differences.


## Kreimer et al. 2017

Comparative review of prediction gene expression in MPRA. Participants used many types of features.
Used assays of epigenetic marks in cell types of interest. Also used predictions of these marks
from deep learning models. The mark predictions were able to take account of the reference and the
alternate allele. The assays were only based on the actual genomic sequence in the cell line studied.
