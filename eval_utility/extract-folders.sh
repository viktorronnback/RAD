# Bash script for extracting cityscapes files from subfolders into the parent folder
# Warning, destructive script, make sure you point the cd call to the correct folder before run

# Change this
cd datasets/cityscapes_unfiltered

for d in */ ; do
    mv $d/* .
    rmdir $d
done
