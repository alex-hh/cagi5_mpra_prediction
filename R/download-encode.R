---
title: "Exploration of CAGI5 data"
output:
  html_document:
    theme: united
    highlight: tango
    toc: true
    toc_float:
      collapsed: false
    fig_width: 10
    fig_height: 6
    fig_caption: true
    df_print: paged
    # mathjax: local
    # self_contained: false
---

```{r render, eval = FALSE, echo = FALSE}
#
# Run this chunk to render this document
rmarkdown::render('download-encode.Rmd')
```

## Setup
```{r loadPkgs}
#!/usr/bin/env Rscript

#
# Script to download ENCODE epigenetic tracks
#


#
# Load packages
#
suppressPackageStartupMessages({
  library(tidyverse)
  library(stringr)
  library(GenomicRanges)
  library(BSgenome)
  library(rtracklayer)
  library(AnnotationHub)
})


#
# Script parameters and config
#
seed <- 37
set.seed(seed)
session <- browserSession("UCSC")
genomeid <- 'hg19'
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
gr <- GRanges(meta$coords[8], seqinfo = genomeseqinfo)
gr


#
# Use AnnotationHub to discover tracks
try_ah <- FALSE
if (try_ah) {

  ah = AnnotationHub()
  # Get every cell line from the meta data
  cell_lines <- unique(unlist(str_split(meta$cell_line, ', ')))
  for (cell_line in cell_lines) {
    print(cell_line)
    clq <- query(ah, cell_line, 'dnase', 'overlap')
    print(clq)
  }

}


#
# Guess at cell line mappings to ENCODE cell lines
#
cell_line_mapping <- list(
  'HEL 92.1.7' = 'GM12878',
  'GBM' = 'Gliobla',
  'SK-MEL-28' = 'Colo829',
  'HaCaT' = 'NHEK',
  'MIN6' = 'PanIslets')  # Not sure at all about this last mapping
get_encode_cell_line <- function(cell_line) {
  if (cell_line %in% names(cell_line_mapping)) {
    cell_line_mapping[[cell_line]]
  } else {
    cell_line
  }
}


#
# Map regions to cell lines
#
region_cell_lines <- as.list(meta$cell_line)
names(region_cell_lines) <- meta$gene
region_cell_lines <- sapply(region_cell_lines, function(cl) str_split(cl, ', '))


#
# Get Duke open chromatin signal table names
#
# Example URL:
# http://genome.ucsc.edu/cgi-bin/hgTables?hgsid=662094631_LCwGKz84uo7rDaGzAQrMHZHoNmx2&clade=mammal&org=Human&db=hg19&hgta_group=regulation&hgta_track=wgEncodeOpenChromDnase&hgta_table=wgEncodeOpenChromDnaseK562BaseOverlapSignalV2&hgta_regionType=genome&position=chr19%3A11199907-11200224&hgta_outputType=wigData&hgta_outFileName=
# track = wgEncodeOpenChromDnase
# table = wgEncodeOpenChromDnaseK562BaseOverlapSignalV2
openChromDnase <- ucscTableQuery(session, track = 'wgEncodeOpenChromDnase', range = gr)
openChromTables <- tableNames(openChromDnase)
openChromTables[grep('mel*BaseOverlapSignal', openChromTables, ignore.case = TRUE)]

get_duke_overlap_track_name <- function(encode_cell_line) {
  tableRegex <- str_c(encode_cell_line, 'BaseOverlapSignal')
  tables <- openChromTables[grep(tableRegex, openChromTables, ignore.case = TRUE)]
  print(tables)
  stopifnot(1 == length(tables))
  data.frame(encode_cell_line = encode_cell_line, tablename = tables, stringsAsFactors = FALSE)
}
get_dnase_track_name <- function(cell_line) {
  # Map the cell line
  encode_cell_line <- get_encode_cell_line(cell_line)
  message(region, '; ', cell_line, '; ', encode_cell_line)
  if (! is.null(encode_cell_line)) {
    tr <- get_duke_overlap_track_name(encode_cell_line)
  } else {
    tr <- data.frame()
  }
  tr$cell_line <- cell_line
  tr
}
get_region_dnase_names <- function(region) {
  tr <- bind_rows(lapply(region_cell_lines[[region]], get_dnase_track_name))
  tr$elem <- region
  tr
}
dnase_track_names <- bind_rows(lapply(names(region_cell_lines), get_region_dnase_names))
get_dnase_track <- function(table_name) {
  message('Getting DNAse track: ', table_name)
  tableName(openChromDnase) <- table_name
  track(openChromDnase)  # a GRanges object
}
dnase_tracks <-
  dnase_track_names %>%
  rowwise() %>%
  do(tr = get_dnase_track(elem = .$elem, cell_line = .$cell_line, .$tablename))
dnase_tracks$tr[[1]]


#
# Extract data from track GRange
#
as_score_df <- function(tr, bp_margin = 20) {
  # Remove NAs
  tr <- tr[! is.na(score(tr)),]
  # Need at least one range
  stopifnot(length(tr))
  # Logic below only works if all ranges are width 1 as they seem to be from UCSC
  stopifnot(all(width(tr)) == 1)
  # Get start and end
  start_pos <- min(start(tr)) - bp_margin
  end_pos <- max(end(tr)) + bp_margin
  # Create data frame to return
  .df <- data.frame(Pos = start_pos:end_pos, Value = 0)
  # Put values and chromosomes in data frame
  idxs <- start(tr) - start_pos + 1
  .df$Value[idxs] <- score(tr)
  .df$chrom[idxs] <- as.vector(chrom(tr))
  .df
}
as_score_df(tr)

query <- ucscTableQuery(session, track = 'wgEncodeUwDgf', table = 'wgEncodeUwDgfK562Sig', range = gr)
query <- ucscTableQuery(session, track = 'wgEncodeUwDgf', range = gr)
tr <- track(query)  # a GRanges object

tableNames(query)

tableName(query) <- 'wgEncodeUwDgfK562Hotspots'
track(query)
tableName(query) <- 'wgEncodeUwDgfK562Pk'
track(query)
tableName(query) <- 'wgEncodeUwDgfK562Sig'
track(query)
tableName(query) <- 'wgEncodeUwDgfK562Raw'
tr <- track(query)
tr$score

trackName(query)
tableName(query)
names(query)
tr <- track(query)  # a GRanges object
width(tr)
width(gr)
ta <- getTable(query)

query <- ucscTableQuery(session, 'wgEncodeDNAseSuper', range = gr)
query <- ucscTableQuery(session, 'wgEncodeOpenChromDnase', range = gr)
query <- ucscTableQuery(session, 'ENC DNase/FAIRE...', range = gr)
track.table <-c('wgEncodeRegMarkH3k4me1', 'wgEncodeBroadHistoneGm12878H3k4me1StdSig')
track.table <-c('wgEncodeUwDgf', 'wgEncodeUwDgfK562Raw')
track.table <-c('dnaShape_MGW', 'dnaShape_2nd_MGW')
query <- ucscTableQuery(session, track.table[1], range = gr, table = track.table[2])
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

track.name <- "wgEncodeUwDgf"
table.name <- "wgEncodeUwDgfK562Hotspots"
e2f3.grange <- GRanges("chr6", IRanges(20400587, 20403336))
tbl.k562.dgf.e2f3 <- getTable(ucscTableQuery(session, track=track.name, range=e2f3.grange, table=table.name))
