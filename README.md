# DICOM Heart Segmentation

## tl;dr
see [segmentation.ipynb](./segmentation.ipynb)

### To run tests:
```bash
pip install -r requirements.txt
./test.sh
```

### Phase 1:
* I have started implementing the solution right from the Part 2, skipping Part 1. In this way, I got a much better understanding of the final purpose of the code required for Part 1 and better view on which design decisions are better suited for solving the task.
* To integrate `parsing.py` into production codebase I've decided to cover most of the code with tests with code coverage report.
* I changed `parse_dicom_file` function from `parsing.py` to return `np.array` instead of `dict` as I found it is more convenient.
* I usually don't add datasets to git, but decided to use it as a fixture for tests.
* Contour parsing is verified by adding tests with toy polygon.
* Also, I manually verified that masks are parsed correctly by overlaying a couple of masks over corresponding images.
* With the current pipeline, the most promising area of improvement might be adding parallel data loading using threads or processes and thread-safe queues.
* I have started with an idea to create 2 data loaders: one for reading data from memory (for datasets small enough to fit into memory, for speeding up data loading) and the other one for reading files from file system. But after some prototyping, I found that this approach adds a lot of unnecessary complexity and that this can be achieved in a much more elegant way by just iterating over the entire dataset and storing it in list: `list(data_loader)`.
* Another thing I noted is `try/except` statement in `parse_dicom_file` in `parsing.py`, I usually would not ignore such errors as it might lead to not always clear behavior when `None` is returned from a function, but decided to leave it as it is as it might be the correct design decision for some cases. 

### Phase 2:
* To meet the requirements of providing o-contours I modified `DataLoader` to return 3-element tuple with image, i-contour, and o-contour. This change breaks the interface of `DataLoader` which might not be suitable for production deployment, to alleviate this one can extend the interface with new methods designed for handling new requirements. A better solution would be to return `dict` instead of tuple which can be extended with any additional data without breaking the interface. To make `DataLoader` suitable for different purposes, it is now can be configured to return only the required data: 
```python
data_loader = DataLoader(..., include_i_contours=False, include_o_contours=True)
```
* For the description of solving the segmentation problem please see [segmentation.ipynb](segmentation.ipynb)
