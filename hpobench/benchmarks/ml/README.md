Each function evalution returns a dictionary with the following information:

```
└───function_value: 1 - accuracy (acc.) on validation set
└───cost: time to fit model + time to evaluate acc. training set + time to evaluate acc. validation set
└───info: dictionary (dict) with miscellaneous information
|   └───train_loss: 1 - accuracy (acc.) on training set
|   └───val_loss: 1 - accuracy (acc.) on validation set
|   └───model_cost: time taken to fit the model
|   └───train_scores: performance on all metrics over the training set (dict)
|   |   └───f1: F1-score   
|   |   └───acc: Accuracy
|   |   └───bal_acc: Balanced accuracy
|   └───train_costs: time taken to compute performance on all metrics over the training set (dict)
|   |   └───f1: F1-score   
|   |   └───acc: Accuracy
|   |   └───bal_acc: Balanced accuracy 
|   └───valid_scores: performance on all metrics over the validation set (dict)
|   |   └───...
|   └───valid_costs: time taken to compute performance on all metrics over the validation set (dict)
|   |   └───...
|   └───test_scores: performance on all metrics over the test set
|   |   └───...
|   └───test_costs: time taken to compute performance on all metrics over the test set (dict)
|   |   └───...
```

*NOTE*: the keys `function_value`, `cost`, `info` need to exist when creating a new objective 
function, while `info` can house any kind of auxilliary information required.