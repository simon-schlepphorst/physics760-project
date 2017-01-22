#!/usr/bin/bash
if [[ ! -e ./Videos ]]
then
	mkdir ./Videos
elif [[ ! -d ./Videos ]]
then
	echo "Videos exists but is not a directory. Please clean up!"
	exit 1
fi

for dir in ./Images/*/
do
	dir=${dir%*/}
	ffmpeg -pattern_type glob -i ${dir}/'*.png' -s 300x300 -vcodec mpeg4 Videos/${dir##*/}.avi
done
