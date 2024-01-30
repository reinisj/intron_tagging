# Image processing pipeline and hit calling code for article submission Reicher, Reiniš, et al., Nature Cell Biology, July 2023

The code will ultimately be deposited and publicly shared in a github repository at https://github.com/reinisj/intron_tagging

**Contents**
- `pipeline_control_script`: used to specify paths to datasets and all processing parameters; generates bash scripts ran by slurm; takes care of parallelization where necessary
- `pipeline_modules`: standalone python scripts which are comprise the image processing pipeline
- `hit_calling`: transforms output of the pipeline into list of hit candidates
- `cellprofiler_pipeline`: input file for CellProfiler - specifies modules used and features to extract
- `conda_environments`: list of installed packages and their versions

