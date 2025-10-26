CSV_PATH=../data/hmda_ne_all_2017.csv
OUTDIR=../reports/hmda_eda2

python hmda_eda.py --input $CSV_PATH --outdir $OUTDIR | tee log.txt

./images_to_pdf.sh $OUTDIR/figures $OUTDIR/all_plots.pdf 
