nf=50
ntr=200

mkdir raw/

for s in 5 8 20 25 30 35; do
  python classical_main.py $nf $ntr $ntr $s "elliptic_preproc.csv"
done
