for ((i=0;i<10;i++)); do
    cd ".\\$i"
    cnt=0
    for file in `ls`; do
        if [ $cnt -lt 10 ]; then
            mv ${file} "$i""0""${cnt}.wav"
        else 
            mv ${file} "$i${cnt}.wav"
        fi
        cnt=$(($cnt+1))
    done
    cd ..
done

echo "Process completed."