#!/bin/bash
START_TIME=$SECONDS

mz_bin="$1"
rt_bin="$2"
spd="$3"
ms_level="$4"
experiment="$5"
if [ $# -ge 1 ]
then
    if [ "$6" == "test" ]
    then
        now='test'
        test=1
    else
        now=$(date +"%Y%m%d_%H%M")
        test=0
    fi
else
    now=$(date +"%Y%m%d_%H%M")
    test=0
fi

cd msml/mzdb2tsv || exit
nprocs=$(nproc --all)
echo "Processing mzdb to tsv using $nprocs processes"
if [ "$ms_level" == "1" ]; then
  find ../../../../resources/"$experiment"/mzdb/"$spd"spd/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
  wait
elif [ "$ms_level" == "2" ]; then
  find ../../../../resources/"$experiment"/mzdb/"$spd"spd/* | parallel -j "$nprocs" JAVA_OPTS="-Djava.library.path=./" ./amm dia_maps_histogram_ms1.sc {} "$mz_bin" "$rt_bin"';' echo "Processing " {}
  wait
fi

# cd ../../../../resources/mzdb/"$spd"spd/"$group" || exit
mkdir -p "../../../../resources/tsv/"$experiment"/mz$mz_bin/rt$rt_bin/"$spd"spd/"
for input in *.tsv
do
    id="${input%.*}"
    echo "Moving $id data"
    mv "$id.tsv" "../../../../resources/tsv/"$experiment"/mz$mz_bin/rt$rt_bin/"$spd"spd/"
done

ELAPSED_TIME=$(($SECONDS - $START_TIME))

eval "echo mzdb2tsv Elapsed time : $(date -ud "@$ELAPSED_TIME" +'$((%s/3600/24)) days %H hr %M min %S sec')"
# cd ../../../../.. || exit
# python3 msml/make_images.py --run_name="$now" --on=all --remove_zeros_on=all --test_run="$test" --mz_bin="$mz_bin" --rt_bin="$rt_bin" --spd="$spd"
