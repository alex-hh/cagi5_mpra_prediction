#!/usr/bin/env Rscript

#
# Script to download ENCODE epigenetic tracks
#


#
# Load packages
#
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(GenomicRanges)
  library(BSgenome)
  library(rtracklayer)
})


#
# Script parameters and config
#
seed <- 37
set.seed(seed)
session <- browserSession("UCSC")
genome(session) <- genomeid
genomeseqinfo <- Seqinfo(genome = genomeid)



#
# Genomic Ranges
#
table_str <- 'gene	coords	phenotype	vector	cell_line	Transf_time	fold_change	construct_size
F9	chrX:138612624-138612923	Hemophilia B	pGL4.11b	HepG2	24	2.6	300
GP1BB	chr22:19710790-19711173	Bernard-Soulier Syndrome	pGL4.11b	HEL 92.1.7	24	22.1	384
HBB 	chr11:5248252-5248438	Thalassemia	pGL4.11b	HEL 92.1.7	24	14.3	187
HBG 	chr11:5271035-5271308	Hereditary persistence of fetal hemoglobin	pGL4.11b	HEL 92.1.7	24	118.1	274
HNF4A (P2)	chr20:42984160-42984444	Maturity-onset diabetes of the young (MODY)	pGL4.11b	HEK293T	24	2.8	285
LDLR	chr19:11199907-11200224 	Familial hypercholesterolemia	pGL4.11b	HepG2	24	110.7	318
MSMB	chr10:51548988-51549578	Prostate cancer	pGL4.11b	HEK293T	24	8.4	591
PKLR	chr1:155271187-155271655	Pyruvate kinase deficiency	pGL4.11b	K562	48	29.4	469
TERT	chr5:1295105-1295362	Various types of cancer	pGL4.11b	HEK293T, GBM	24	231.8	258
IRF4	chr6:396143-396593 	Human pigmentation	pGL4.23	SK-MEL-28	24	44.5	451
IRF6	chr1:209989135-209989734 	Cleft lip	pGL4.23	HaCaT	24	17.0	600
MYC	chr8:128413074-128413673 	Various types of cancer (rs6983267)	pGL4.23	HEK293T	32, 20nM LiCl added after 24hr	0.8	600
SORT1	chr1:109817274-109817873	Plasma low-density lipoprotein cholesterol & myocardial infraction	pGL4.23	HepG2	24	235.3	600
ZFAND3 	chr6:37775276-37775853 	Type 2 diabetes	pGL4.23	MIN6	24	14.3	578'
meta <- readr::read_tsv(table_str)
meta
meta$coords
# gr <- Reduce(union, lapply(coords, function(.c) as(.c, 'GRanges')))
gr <- GRanges(meta$coords[4], seqinfo = genomeseqinfo)
gr


#
# Example from rtracklayer package vignette
#
trackNames(session)
names(trackNames(session))
tracks_of_interest <- c(
  'wgEncodeReg',
  'wgEncodeDNAseSuper',
  '',
query <- ucscTableQuery(session, 'wgEncodeDNAseSuper', range = gr)
query <- ucscTableQuery(session, 'wgEncodeOpenChromDnase', range = gr)
query <- ucscTableQuery(session, 'ENC DNase/FAIRE...', range = gr)
tableNames(query)
tableName(query) <- 'wgEncodeOpenChromDnaseK562BaseOverlapSignalV2'
e2f3.grange <- GRanges("chr6", IRanges(20400587, 20403336))
width(e2f3.grange)
track.table <-c('wgEncodeRegMarkH3k4me1', 'wgEncodeBroadHistoneGm12878H3k4me1StdSig')
track.table <-c('wgEncodeUwDgf', 'wgEncodeUwDgfK562Raw')
track.table <-c('dnaShape_MGW', 'dnaShape_2nd_MGW')
query <- ucscTableQuery(session, track.table[1], range = e2f3.grange, table = track.table[2])
## list the table names
# tableNames(query)
query
## retrieve the track data
tr <- track(query)  # a GRanges object
tr
tr$score
sum(width(tr))
width(gr)
intersect(tr, gr)
