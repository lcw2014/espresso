#!/bin/bash

data_dir=$1
target_data_dir=$2

echo 'generate lexicon for Korean'

# text normalization
cat $target_data_dir/text.txt | sed -e s/' '/'\n'/g | LC_COLLATE='utf-8' sort -u | grep -v '^$' | grep -v '<' | grep -v '>' | grep -v '_' | grep -v '\.' | grep -v '^▁$' | grep -v '#' | grep -v '%' > $data_dir/words.txt

cat $data_dir/words.txt | sed s/'▁'/''/g > words

# Decompose Korean words & prepare lexicon.txt
python3 local/gen_lexicon_HD.py

echo '!SIL	SIL' > $data_dir/lexicon.txt
echo '<SPOKEN_NOISE>	SPN' >> $data_dir/lexicon.txt
echo '<UNK>	SIL' >> $data_dir/lexicon.txt

paste $data_dir/words.txt decomposed_phone.txt >> $data_dir/lexicon.txt

rm -f words
rm -f decomposed_phone.txt
