# DICOM Data Loader
### work in progress, super raw readme

can be improved with parallel loading with scheduling and queues
started from in_memory but then understood this is no use
i usually dont add datasets to git
outline true false
not sure if ignoring error is okay

i changed `parse_dicom_file` function to return `np.array` instead of `dict` as i found it is not very useful

0.0 value checks

InvalidDicomError

masks include polygon or not

* To integrate parsing.py into production codebase i've decided to cover most of the code with tests
* I usually dont add datasets to git, but decided to use it as fixture for tests
* Contour parsing is verified by adding test with toy polygon
* With the current pipeline one of the best areas of 
time investment would be to add parallel data loading using     
queue 

requirements.txt