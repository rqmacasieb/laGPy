# laGPy - Python implementation of local approximate GP (laGP)
What is local approximate GP?
-----------------------------------------------
This tool is largely based on the methods of Gramacy (2016). Local approximate GP (laGP) approximates GP regression by sequentially selecting local designs, which is a subset of the whole training dataset, with respect to some untried location in the input space and make output prediction at that loaction.

A complete documentation about laGP and an associated R package can be found in https://bobby.gramacy.com/r_packages/laGP/

Installation
-----------------------------------------------
laGPy requires **Python** 3.8 (or higher). To install laGPy:
  
    pip install lagpy

Disclaimer
-----------------------------------------------
This Python implementation of laGP (Local Approximate Gaussian Process) was independently developed from the original R package. While this software implements the algorithms and methods described in Gramacy (2016), it is preliminary or provisional and is subject to revisions. In its current form, this package only implements the 'nearest neighbour' and 'active learning Cohn' techniques for greedy selection of local designs. Additional functionalities that was not in the original R package has been added such as building a GP using the full training dataset and saving such GP model as pickle file that can be called and executed for later purposes.

This implementation is based on version 1.5-8 of the original laGP R package. Any features, improvements, or bug fixes introduced in versions after 1.5-8 of the original R package may not be incorporated in this Python version.

While the implementation is original, the intellectual approach follows the published methodology of the original authors. This package is distributed under the terms of the GNU Lesser General Public License (LGPL) version 2 or later, consistent with the licensing of the original work.

References
-----------------------------------------------
Robert B. Gramacy & Daniel W. Apley (2015) Local Gaussian Process Approximation for Large Computer Experiments, Journal of Computational and Graphical Statistics, 24:2, 561-578, https://doi.org/10.1080/10618600.2014.914442

Gramacy, R. B. (2016). laGP: Large-scale spatial modeling via local approximate Gaussian processes in R. Journal ofStatistical Software, 72(1), 1–46. https://doi.org/10.18637/jss.v072.i01
