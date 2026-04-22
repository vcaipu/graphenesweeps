python masterscript.py --strain=27.5 --guess=28
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=27 --guess=20
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=24 --guess=22
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=26 --guess=24
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete
python masterscript.py --strain=28 --guess=26
find . -name "*.wfc*" -type f -delete && find . -name "*.hdf5" -type f -delete