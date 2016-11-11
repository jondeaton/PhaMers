#!/bin/bash
#SBATCH --job-name=phamer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=long
#SBATCH --mem-per-cpu=8000
#SBATCH --mem=24000
#SBATCH --time=12:00:00
#SBATCH --error=/local10G/jdeaton/Phamer/outputs/error-%A.out
#SBATCH --output=/local10G/jdeaton/Phamer/outputs/log-%A.out

scratch=$LOCAL_SATA
now=`date +%Y-%m-%d.%H.%M.%S`

do_taxonomy=true
do_cross_validation=true
do_phamer=true
do_analysis=true

run_bijah_road_side4=false
run_lower_geyser_basin=true

# Locations
home=~
python=$home"/anaconda/bin/python"
phamer_directory=$home"/PhaMers"
script_directory=$phamer_directory"/scripts"
data_directory=$home"/data"
datasets_directory=$home"/datasets"

# Scripts
phage_taxonomy=$script_directory"/feature_taxonomy.py"
cross_validation=$script_directory"/cross_validate.py"
phamer=$script_directory"/phamer.py"
analysis=$script_directory"/analysis.py"

# Inputs
bijah_road_side4=$datasets_directory"/bijah_road_side4"
lower_geyser_basin=$datasets_directory"/lower_geyser_basin"

# Outputs
taxonomy_plots=$phamer_directory"/outputs/taxonomy"
cross_validation_out=$phamer_directory"/outputs/cross_validation"

# Dataset Specific Locations
if $run_lower_geyser_basin
    then
        input_directory=$lower_geyser_basin
elif $run_bijah_road_side4
    then
        input_directory=$bijah_road_side4
fi

# Phage Taxonomy (Figure 1)
if $do_taxonomy
    then
        echo "====== Phage Taxonomy ======"
        all_phage=$data_directory"/all_phage_genomes.csv"
        phage_features=$data_directory"/reference_features/positive_features.csv"
        bacteria_features=$data_directory"/reference_features/bacteria_features.csv"
        phage_lineages=$data_directory"/phage_lineages.txt"
        phage_tsne_file=$phamer_directory"/outputs/taxonomy/tsne_data.csv"
        $python $phage_taxonomy -fasta $all_phage -features $phage_features -lin $phage_lineages -tsne $phage_tsne_file -out $taxonomy_plots --debug --do_tsne
fi

# Cross Validation (Figure 2)
if $do_cross_validation
    then
        echo "====== Cross Validation ======"
        phage_features=$data_directory"/reference_features/positive_features.csv"
        bacteria_features=$data_directory"/reference_features/negative_features.csv"
        cuts_directory=$data_directory"/cut_features/cut_4mer_counts"
        $python $cross_validation -pf $phage_features -nf $bacteria_features -out $cross_validation_out -N 20 -a --debug --test_all
fi

# Phamer Scoring (Figure 3)
if $do_phamer
    then
        echo "====== Phamer Run ======"
        $python $phamer -in $input_directory --data_directory $data_directory --do_tsne --debug
fi

# Analysis (Figure 4)
if $do_analysis
    then
        echo "====== Analysis ======"
        $python $analysis -in $input_directory --data_directory $data_directory --debug
fi
