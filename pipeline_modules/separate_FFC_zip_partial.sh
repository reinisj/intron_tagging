measurement_name=$(basename "$measurement")
measurement_name_FFC="$measurement_name"_FFC

# separate FFC-corrected images
n_raw=$(find $measurement/Images -type f -name "r*" | wc -l)
n_flex=$(find $measurement/Images -type f -name "flex*" | wc -l)
echo $measurement_name

# confirm that the FFC-correction was fully completed
if [ $n_raw -ne $n_flex ]; then
    echo "The number of images ($n_raw) does not match the number of FFC-corrected images ($n_flex). Exiting"
    exit 1
else
    echo "$n_raw raw images, $n_flex FFC-corrected images"
fi

# confirm that we are not overwriting any existing files  
if [ -d $measurement_FFC/Images_FFC ]; then
    nfiles=$(find $measurement_FFC/Images_FFC -type f | wc -l)
    if [ $nfiles -gt 0 ]; then
        echo "Directory $measurement_FFC/Images_FFC already exists with $nfiles files. Not overwriting and exiting."
        exit 1
    fi
fi

# create new folder for the FFC-corrected measurement
mkdir -p $measurement_FFC
# create subfolders - copy auxiliary files (index, assay layout) and make a folder for the images
cp -r $measurement/Assaylayout $measurement/FFC_Profile $measurement_FFC/
mkdir -p $measurement_FFC/Images_FFC

# move or copy the FFC-corrected images
if [ $delete_flex == "True" ]; then
    ls $measurement/Images/flex/ | xargs -I{} mv $measurement/Images/flex/{} $measurement_FFC/Images_FFC
else
    ls $measurement/Images/flex/ | xargs -I{} cp $measurement/Images/flex/{} $measurement_FFC/Images_FFC
fi

# remove the flex_ prefix, but only in the newly copied files
ls $measurement_FFC/Images_FFC | xargs -I{} $rename_script "flex_" "" $measurement_FFC/Images_FFC/{}
# reorder (rename) channels
i=1
#for ch in ${channels_order[@]}; do rename -- "-ch$ch" "-TEMP$i" $measurement_FFC/Images_FFC/*; ((i++)); done

for ch in ${channels_order[@]}
do
    ls $measurement_FFC/Images_FFC/ | grep ch$ch | xargs -I{} $rename_script -- "-ch$ch" "-TEMP$i" $measurement_FFC/Images_FFC/{}
    ((i++))
done

ls $measurement_FFC/Images_FFC | xargs -I{} $rename_script "TEMP" "ch" $measurement_FFC/Images_FFC/{}

# remove the (now empty) folder with the FFC-corrected images - double check that there are no more files
if [ $delete_flex == "True" ]; then
    # this variable should now be zero
    n_flex=$(find $measurement/Images -type f -name "flex*" | wc -l)
    if [ $n_flex -eq 0 ]; then
        echo "Deleting folder $measurement/Images/flex now with $n_flex files"
        rm -r $measurement/Images/flex
    # don't delete the data if there was any problem with moving the data
    else
        echo "[delete original flex folder] Error: the folder $measurement/Images/flex still has $n_flex files. Exiting."
        exit 1
    fi
fi

# zip non-corrected measurement - if for some reason FFC-corrected files were not correctly moved, they are not included in the archive
echo "Zipping $measurement_name (pre-FFC)"
if [ $zip != "False" ]
then
    if [ -f $zip ]; then echo "Zipped file $zip already exists. Not overwriting."; else cd $measurement/..; zip -qr9 $zip $measurement_name -x "$measurement_name/Images/flex/*"; fi
fi

# zip corrected measurement - if asked for
if [ $zip_FFC != "False" ]
then
    echo "Zipping $measurement_name (post-FFC)"
    # this should match n_raw now
    nfiles=$(find $measurement_FFC/Images_FFC -type f -name "r*" | wc -l)

    if [ -f $zip_FFC ]
    then
        echo "Zipped file $zip_FFC already exists. Not overwriting."
    elif [ $nfiles -ne $n_raw ]; then
        echo "[zip] Error: the number of separated images ($nfiles) does not match the original number of images in the dataset ($n_raw). Exiting"
        exit 1
    else
         cd $measurement_FFC/..
         zip -qr9 $zip_FFC "$measurement_name_FFC/Images_FFC" "$measurement_name_FFC/Assaylayout" "$measurement_name_FFC/FFC_Profile"
    fi
fi

echo; date
sstat -j $SLURM_JOB_ID.batch --format=JobID,Node,MaxVMSize
