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
This Python implementation of laGP (Local Approximate Gaussian Process) is an independent project not affiliated with the original R package (version 1.5-8). It implements the core concepts and algorithms from Gramacy (2016) but was developed separately without input from the original developers of the R package. As such, it may not include all features or recent updates available in the R implementation.

In its current form, this package implements two key techniques for local design selection: 'nearest neighbour' and 'active learning Cohn'. It also offers additional features not found in the original R package, including the ability to build a GP using the full training dataset and save models as pickle files for later use and easier integration in other python-based tools and models that may need such GP models. This implementation is still under development and may receive updates and additional features in future releases.

This package is distributed under the terms of the GNU Lesser General Public License (LGPL) version 3 or later, consistent with the licensing of the original work.

References
-----------------------------------------------
Robert B. Gramacy & Daniel W. Apley (2015) Local Gaussian Process Approximation for Large Computer Experiments, Journal of Computational and Graphical Statistics, 24:2, 561-578, https://doi.org/10.1080/10618600.2014.914442

Gramacy, R. B. (2016). laGP: Large-scale spatial modeling via local approximate Gaussian processes in R. Journal ofStatistical Software, 72(1), 1–46. https://doi.org/10.18637/jss.v072.i01
