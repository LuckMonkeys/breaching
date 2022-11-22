a=1
b=2
for start in {1..8}
do
    echo $b 
    echo $a &
    sleep 5
    # echo $GPU
done