# Process-Mining-Group22
Process Mining course project\
*Made by group 22*

This tool can be used to analyse the performance of 2 different methods using the BPI challenge datasets as the case study.\
To run the tool, run the main.py file. This will run all necessary files.

To give more options, 3 different optional arguments can be provided when running the main file. These are:

```angular2html
--dataset      This can be used to select the wanted BPI challenge dataset. Default is 2012, options are [2012, 2017]

--generate     This can be used to select whether to generate the data from the starting CSV file
               or whether the data should be read from a file which has already had the preprocessing done. 
               Default is 0 (the data should be read), options are [0: read the data, 1: generate the data]
               
--plots        This can be used to indicate whether the plots, which are on the poster should be generated.
               This might take a while. Default is 0 (they should not be generated), 
               options are [0: the plots should **NOT** be generated, 1: the plots should be generated] 
```
Example usage, after the repository is put to the Process-Mining-Group22 folder:
```angular2html 
py main.py --dataset 2012 --generate 0 --plots 0 ```
