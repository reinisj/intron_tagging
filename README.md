# Pooled multicolour tagging for visualizing subcellular protein dynamics

Image processing pipeline and detecting drug-induced changes to proteins for article [**Reicher&ast;, Reiniš&ast;, et al. Nature Cell Biology, 2024.**](https://www.nature.com/articles/s41556-024-01407-w)

<p align="center"><img src="graphical_abstract2.png "/></p>

**Contents**
- `pipeline_control_script`: used to control the pipeline run, specifies paths to datasets and all processing parameters; generates bash scripts ran by slurm; takes care of parallelization where necessary
- `pipeline_modules`: standalone python scripts invoked by individual parts of the pipeline
- `hit_calling`: transforms output of the pipeline into a list of hit candidates
- `cellprofiler_pipeline`: input file for CellProfiler - specifies modules used and features to extract
- `model_training`: training of the random forest models for clone discrimination
- `conda_environments`: list of packages and versions
  
