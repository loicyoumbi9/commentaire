#!/bin/bash
a=`ls`
i=1
for file in $a
do
	if [ -d "$file" ]; then
		find . -type d -exec bash -c 'cd "{}" ; find . -maxdepth 1 -type f -name "*.java" -printf "%f\n" | sort -n | xargs cat > makefile.txt; test -s makefile.txt || rm makefile.txt' \;
	
	fi
	mv */* .
done
