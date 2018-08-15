FOLDER=$1
for nb in "$FOLDER/*.ipynb" 
do
  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $nb
done
