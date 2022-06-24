for f in $(ls $1*.kb); do
	echo "File > $f"
	echo "Entities: $(cat $f | grep -P ".+\t.+\t.+\t.+"|wc)"
	echo "Overlap: $(grep -Fxf  /data/cgonzale/graphvite_data/wikidata5m_ids_existing.txt <(cat $f |grep -P ".+\t.+\t.+\t.+"|cut -f3) |wc)"
done
