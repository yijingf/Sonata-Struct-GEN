#!/bin/bash
#
# Description: Converts .krn files under krn-files/composer to MusicXML (.xml) using humextra tools.
# Usage: ./krn2xml.sh /path/to/krn-files
#
# Example:
#   ./krn2xml.sh ../
#
# Notes:
# - Requires humextra (https://github.com/craigsapp/humextra) to be installed and in PATH.
# - Output .xml files will be saved under <krn-files/composer> directories.

if [ -z $1 ]; then
  echo "Specify the path to .krn files" >&2
  exit 1
fi

root_dir=$1

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



