# ffselect

### Fast feature subset selection library

This algorithm performs feature subset selection in O(n log n) time or faster

## Usage

```python
MinSubsetSelection(data, target, fit_function, features, loss=True, interactive=True)

"""
Minimal feature subset selection algorithm
    data: Input data to pass to the fitting function
    target: Target parameter feature name
    fit_function: Callable function that fits the model and returns R^2/loss
    features: List of feature names
    loss: Set to true (by default) if the fitting function returns loss, R^2 otherwise
    interactive: Print output (default True)
    
    return: Tuple with the resulting R^2/loss and the list of features
"""
```

Please view [subset.ipynb](https://github.com/enaix/ffselect/blob/main/examples/subset.ipynb) for the complete example

## Installation

`pip3 install ffselect`