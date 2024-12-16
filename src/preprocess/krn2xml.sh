#!/bin/bash
#
# Convert .krn in `path/to/krn-dataset/krn` to music .xml in `path/to/krn-dataset/mxml` using humextra (https://github.com/craigsapp/humextra).
# Usage: bash krn2xml.sh path/to/krn-dataset
#

root_dir=$1 #"../../sonata-dataset"

declare -a composers=("mozart" "beethoven" "haydn" "scarlatti")

# Iterate through 4 composers
for composer in "${composers[@]}"; do
    echo ""
    echo "$composer"
    echo ""

    in_dir="$root_dir/krn/$composer"
    out_dir="$root_dir/mxml/$composer"

    mkdir -p $out_dir

    for input_file in $in_dir/*.krn; do

        file=$(echo "$input_file" | awk -F'/' '{print $NF}')
        file=$(echo "$file" | awk -F'.' '{print $1}')
        output_file="$out_dir/$file.xml"

        echo $input_file
        ./humextra/bin/hum2xml $input_file > $output_file
        echo $output_file

        echo ""

    done
done



