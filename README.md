## Code for [Confidence sequences for sampling without replacement](https://arxiv.org/pdf/2006.04347.pdf)

### Code layout

```
├── cswor
│   ├── cswor.py                        # Main code for this project. Contains essential
│                                       # functions like BBHG_confseq, hoeffding_wor, etc.
│
│   ├── misc.py                         # Non-critical functions that are still useful enough
│                                       # to keep in the project.
│
│   └── utils.py                        # Primarily for helper functions for cswor.py and misc.py
│                                       
├── figures                             # Figures for the paper or playing around.
│                                       # Typically in .pdf format and generated
│                                       # by ipynb files in ./notebooks folder.
│
├── notebooks
│   └── plots_for_paper.ipynb           # Plots for the paper, and also contains
│                                       # good examples for getting acquainted 
│                                       # with the code 
```

### Dependencies
* Python 3
  * matplotlib
  * numpy
  * scipy
