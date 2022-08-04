# TT-CME

Tensor Train for the Chemical Master Equation. 
This repository implements the paper ["Tensor-train approximation of the chemical master equation and its application for parameter inference"](https://aip.scitation.org/doi/10.1063/5.0045521) on top of the [torchTT](https://github.com/ion-g-ion/torchTT) package ( **upgraded version of [this repository](https://github.com/ion-g-ion/paper-cme-tt)** )

## Installation

### Requirements

 * `pytorch>=1.7`
 * `numpy>=1.18`
 * `scipy`
 * [`torchtt`](https://github.com/ion-g-ion/torchtt)
 * `opt_einsum`
 * `matplotlib`
 * `numba`
### Using pip

```
pip install git+https://github.com/ion-g-ion/tt-cme
```

## Packages


Sub-modules:
 * `TTCME.TimeIntegrator`: Tensor train integrator for linear ODEs in the TT format (implements tAMEn)
 * `TTCME.basis`: Implements the basic univariate bases.
 * `TTCME.pdf`: This contains the basic probability density function pdfTT represented using tensor product basis and TT DoFs.
 * `TTCME.ttcme`: This module implements the ChemicalReaction class as well as the ReactionSystem class.

The documentation can be found [here](https://ion-g-ion.github.io/tt-cme/TTCME/index.html) and is generated using `pdoc3` with:

```
pdoc3 --html tt_iga -o docs/ --config latex_math=True --force
```
## Scripts and examples:

In [this](./examples/) folder a couple of examples are presented:
* [simple_gene.ipynb](./examples/simple_gene.ipynb) basic 2d simple gene expression model.
* [bistable_toggle_model.ipynb](./examples/bistable_toggle_model.ipynb) bistable toggle switch model (bimodal solution).
* [seir_model.ipynb](./examples/seir_model.ipynb) solving the 4d SEIR model.
* [simple_gene_convergence](./examples/simple_gene_convergence.ipynb) convergence study for the simple gene expression model with no parameter.
* [seir_filtering.ipynb](./examples/seir_filtering.ipynb) filtering and smoothing for the 4d SEIR model.
* [simple_gene_param.ipynb](./examples/simple_gene_param.ipynb) parameter dependent simple gene expression model.
* [simple_gene_param_inference.ipynb](./examples/simple_gene_param_inference.ipynb) the parameter inference for the simple gene expression model.
* [3stage_param_inference.ipynb](./examples/3stage_param_inference.ipynb) the parameter inference for the 3 stage gene expression model.
* [SEIQR_param_inference.ipynb](./examples/SEIQR_param_inference.ipynb) the parameter inference for the SEIQR model.


## Author

Ion Gabriel Ion, ion.ion.gabriel@gmail.com
