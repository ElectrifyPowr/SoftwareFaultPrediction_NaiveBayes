# Software Fault Prediction with Naive Bayes
## To make your test phase more efficient

This a basic Software Fault Prediction (SFP) program in <b>Java</b>.<br>
It uses a <b>Naive Bayes Classifier</b> for predicting software faults.<br><br>

A custom, rather small dataset is used (<b>dataset.txt</b>).<br>
The structure is as follows:<br>
module | f1          | f2            | ...           | fn
------------- | ------------ | ------------- | ------------- | -------------
1 | m1\_f1          | m1\_f2            | ...           | m1\_fn
2 | m2\_f1          | m2\_f2            | ...           | m2\_fn
... | ...          | ...            | ...           | ...
k | mk\_f1          | mk\_f2            | ...           | mk\_fn

<br><br>

The used equations can be seen below.<br><br>


<img src="https://latex.codecogs.com/svg.latex?\large&space;classify(f_1,..,f_n)&space;=&space;argmax\&space;p(C=c)&space;\prod_{i=1}^n&space;p(F_i=f_i|C=c)" title="\large classify(f_1,..,f_n) = argmax\ p(C=c) \prod_{i=1}^n p(F_i=f_i|C=c)" />
<br>
**Equation 1: Naive Bayes Classifier**

<img src="https://latex.codecogs.com/svg.latex?\large&space;\sigma&space;=&space;\sqrt{\frac{\sum_{i=1}^{n}&space;(x_i&space;-&space;\overline{x})^2}&space;{n-1}}" title="\large \sigma = \sqrt{\frac{\sum_{i=1}^{n} (x_i - \overline{x})^2} {n-1}}" />
<br>
**Equation 2: Standard Deviation**

<img src="https://latex.codecogs.com/svg.latex?\large&space;f_g(x)&space;=&space;\frac{1}{\sqrt{2\pi\sigma^2}}&space;\&space;exp(\frac{-(x-\overline{x})^2}{2\sigma^2})" title="\large f_g(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \ exp(\frac{-(x-\overline{x})^2}{2\sigma^2})" />
<br>
**Equation 3: Gaussian Distribution Function**

<img src="https://latex.codecogs.com/svg.latex?\large&space;\overline{x}&space;=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_i" title="\large \overline{x} = \frac{1}{n} \sum_{i=1}^{n} x_i" />
<br>
<b>Equation 4: Arithmetic Mean</b>

<img src="https://latex.codecogs.com/svg.latex?\large&space;p(C=i)&space;=&space;\frac{n_i}{N}" title="\large p(C=i) = \frac{n_i}{N}" /><br>
**Equation 5: Category Specific Set Size**


###Run from command line:<br>
First compile the project:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ```javac model/SFPModel.java``` 
<br>
Then run it:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ```java -cp . module/SFPModel```


## Notes

A second dataset file is included
Uses PROMISE dataset

