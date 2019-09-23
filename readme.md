# Age Estimation - `Regression_combination` branch

<!-- TOC -->

- [Age Estimation - `Regression_combination` branch](#age-estimation---regressioncombination-branch)
  - [Git Network graph](#git-network-graph)
  - [to do](#to-do)
  - [experiments](#experiments)
    - [results](#results)
  - [references](#references)

<!-- /TOC -->


## Git Network graph

```sh
age-estimation
        |----------- regression_combination
```

## to do

- [ ] test function, 
$l_{n}=-w_{n}\left[t_{n} \cdot \log \sigma\left(x_{n}\right)+\left(1-t_{n}\right) \cdot \log \left(1-\sigma\left(x_{n}\right)\right)\right]$
- [ ] elaborate the method of the face detection and face alignment, currently I can not remember.
- [ ] continue implementing the bar chart drawing function, the reference link is [here](https://pythonspot.com/matplotlib-bar-chart/)
- [ ] implement epsilon $\epsilon$ error function
- [x] create new branch - regression_loss_combination, to experiments all possible experiments, $15 = 2^{4}-1$
- [ ] **solve the Gaussian loss calculation in the age classification branch**
- [x] experimented the **label smoothing** idea, it improves.
- [x] ~~Gaussian Loss function does not work, no gradient problem~~
- [ ] ~~clean the multitask learning source code to multi-loss age estimation task,~~  
- [x] ~~reference ni xingyang's repository.~~


## experiments

### results




```sh
# run experiments on the `regression_combination`
$ python main.py  
```




## references

