#!/bin/bash
# This script is for running all Phamer related scripts

clear
ls
clear
echo "============================== RUN CALL =============================="
date

pt=False
xv=False
pm=True
pa=True

bijah=True
LGB=False

n_fold=5
method="combo"

# Locations
home=~
phamer_directory=$home"/Dropbox/Documents/research/phamer"
local_data=$home"/Documents/research/local_data"
script_directory=$phamer_directory"/scripts"

# Scripts
phamer=$script_directory"/phamer.py"
cross_validation=$script_directory"/cross_validate.py"
post_analysis=$script_directory"/post_analysis.py"
phage_taxonomy=$script_directory"/phage_taxonomy.py"

# Inputs
bijahRS4=$phamer_directory"/inputs/super_contigs.BijahRoadSide4.fasta"
LowerGeyserBasin=$phamer_directory"/inputs/super_contigs.LowerGeyserBasin_Unknown43.3.fasta"

# Data
all_phage=$phamer_directory"/data/all_phage_genomes.fasta"
phage_lineages=$phamer_directory"/data/phage_lineages.txt"
phage_kmers=$phamer_directory"/data/phage_4mers.csv"
bacteria_kmers=$phamer_directory"/data/bacteria_2_4mers.csv"
cuts_file=$local_data"/cuts"

# Outputs
taxonomy_plots=$phamer_directory"/outputs/taxonomy_plots"
CV_out=$phamer_directory"/outputs/cross_validation"

# VirSorter
bijahRS4_VS=$local_data"/VirSorter_1.0.3_bijahRS4-2016-03-28-18-53-29.7"
LGB_VS=$local_data"/VirSorter_1.0.3_LGB-2016-04-22-00-31-16.2"

# Post Analysis Output
bjRS4_post_out=$local_data"/post_analysis_bjrs4"
LGB_post_out=$local_data"/post_analysis_LGB"

# Dataset Specific Locations
if $LGB
    then
        in=$LowerGeyserBasin
        kmers_in=$phamer_directory"/outputs/LGB_4mer.csv"
        out=$phamer_directory"/outputs/phamer_out_LGB"
        tsne_out="."
        contig_kmers=$phamer_directory"/outputs/LGB_4mer.csv"
        destination=$local_data"/post_analysis_LGB"
        vs_out=$LGB_VS
        IMG=$phamer_directory"/outputs/IMG_LGB"
        dataset="Lower_Geyser_Basin"
fi

if $bijah
    then
        in=$bijahRS4
        kmers_in=$phamer_directory"/outputs/bijahRS4_4mers.csv"
        out=$phamer_directory"/outputs/phamer_out_2016-05-28.00.14.00"
        tsne_out=$out"/tsne_out.csv"
        contig_kmers=$phamer_directory"/outputs/bijahRS4_4mers.csv"
        destination=$local_data"/post_analysis_bjrs4"
        vs_out=$bijahRS4_VS
        IMG=$phamer_directory"/outputs/IMG_bijahRS4"
        dataset="Bijah_Road_Side_4"
fi

echo "DataSet: "$dataset
summary=$out"/summary.txt"

# Phage Taxonomy (Figure 1)
if $pt
    then
        echo "====== Phage Taxonomy ======"
        phage_tsne_file=$phamer_directory"/outputs/phage_tsne.csv"
        python $phage_taxonomy -f $all_phage -pk $phage_kmers -l $phage_lineages -t $phage_tsne_file -out $taxonomy_plots -v
fi

# Cross Validation (Figure 2)
if $xv
    then
        echo "====== Cross Validation ======"
        python $cross_validation -p $phage_kmers -n $bacteria_kmers -N $n_fold -m $method -cuts $cuts_file -v -out $CV_out
fi

# Phamer Scoring (Figure 3)
if $pm
    then
        echo "====== Phamer Run ======"
        python $phamer -in $in -pk $phage_kmers -nk $bacteria_kmers -k 4 -l 5000 -out $out -m $method -v
fi

# Post Analysis (Figure 4)
if $pa
    then
        echo "====== Post Analysis ======"
        python  $post_analysis -p $summary -vs $vs_out -img $IMG -t $tsne_out -pk $phage_kmers -ck $contig_kmers -c $in -l $phage_lineages -d $destination -ds $dataset -v
fi