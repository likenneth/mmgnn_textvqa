# text-vqa
This repo is developed based on Pythia, Facebook's codebase for 
VQA problems. 

Specifications of components of this project:
+ Makefile can help deploy training in screens in one command
+ a data directory is gitignored, containing data
+ a save directory is gitignored, containing results, 
debug and visualization materials, in which
    + `<task_name_l>\_<task_name_s>\_<model_name>` are 
    the configured by Pythia.
    + `tb` are used to store information for tensor board.
    + all other directories should be incrementally 
    added to store different information. 
    
On code_names and how to record:
+ one model should concord with one row in Onenote and one page in 
    Excel, and only record beat in Onenote; variations of it 
    should be recorded in Excel.
