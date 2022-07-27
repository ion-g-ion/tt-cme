# TT-CME

Tensor Train for the Chemical Master Equation. 
This repository implements the paper ["Tensor-train approximation of the chemical master equation and its application for parameter inference"](https://aip.scitation.org/doi/10.1063/5.0045521) on top of the [torchTT](https://github.com/ion-g-ion/torchTT) package (mirror of [this repository](https://github.com/ion-g-ion/paper-cme-tt))

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

## Scripts and examples:

In the [examples](./examples/) folder a couple of examples are presented. 

* [simple_gene_param_inference.ipynb](./examples/simple_gene_param_inference.ipynb) the parameter inference for the simple gene expression model.
* [SEIQR_param_inference.ipynb](./examples/SEIQR_param_inference.ipynb) the parameter inference for the SEIQR model.

The documentation is generated using `pdoc3` with:

```
pdoc3 --html tt_iga -o docs/ --config latex_math=True --force
```

## Author

Ion Gabriel Ion, ion.ion.gabriel@gmail.com