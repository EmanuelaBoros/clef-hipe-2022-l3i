for f in $(ls $1*.kb); do
	echo "File > $f"
	python kb/time_match.py --cfile $f

done