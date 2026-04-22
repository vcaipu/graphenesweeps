python masterscript.py --strain=25.9 --guess=26
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=25.8 --guess=26
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=25.7 --guess=25.8
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=25.6 --guess=25.7
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete