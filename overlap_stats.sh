for f in $(ls $1*.kb); do
	echo "File > $f"
	echo "Entities: $(cat $f | grep $".*\t.*\t*\t"|wc|cut -f2)"
	echo "Overlap: $(grep -Fxf  /data/cgonzale/graphvite_data/wikidata5m_ids_existing.txt <(cat $f |grep $".*\t.*\t*\t"|cut -f3) |wc)"
done
