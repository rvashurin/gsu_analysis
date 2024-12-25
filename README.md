To produce PRR tables call `python produce_tables.py`. If `--do_sample` flag is present tables will be generated with PRRs calculated against sampled generation metrics, otherwise greedy responses will be used.

`python analyze.py` will produce a folder `eda/` that contains, for every model and dataset, a list of directories, one for each pairwise combination of UE Methods. Each directory contains reports on inputs for which methods disagree on rank of the input. 

For both scripts to work, full set of managers must be places in the `sample_metric_mans/log_exp` subdirectory.
