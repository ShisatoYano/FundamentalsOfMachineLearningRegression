# FundamentalsOfMachineLearningRegression
---
My Studying Log of fundamentals about Machine Learning Regression.

## Table of contents
---
<!-- TOC -->

- [FundamentalsOfMachineLearningRegression](#fundamentalsofmachinelearningregression)
    - [Table of contents](#table-of-contents)
    - [Introduction](#introduction)
    - [Author](#author)
    - [Linear model with 1 dimensional input](#linear-model-with-1-dimensional-input)
        - [Input data: Age](#input-data-age)
        - [Target data: Height](#target-data-height)
        - [Data generation](#data-generation)
        - [Linear model definition](#linear-model-definition)
        - [Gradient method](#gradient-method)

<!-- /TOC -->

## Introduction
---
This is my studying log about machine learning, regression. I referred to a following book.  
[Pythonで動かして学ぶ! あたらしい機械学習の教科書](https://www.amazon.co.jp/Python%E3%81%A7%E5%8B%95%E3%81%8B%E3%81%97%E3%81%A6%E5%AD%A6%E3%81%B6%EF%BC%81-%E3%81%82%E3%81%9F%E3%82%89%E3%81%97%E3%81%84%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%81%AE%E6%95%99%E7%A7%91%E6%9B%B8-%E4%BC%8A%E8%97%A4-%E7%9C%9F-ebook/dp/B078767Y56/ref=sr_1_12?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&keywords=%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92&qid=1556694357&s=gateway&sr=8-12)
I extracted some important points and some related sample python codes and wrote them as memo in this article.  

## Author
---
[Makoto Ito](https://researchmap.jp/itomakoto/)

## Linear model with 1 dimensional input
---
### Input data: Age  
$$\bm{x} = \left(
    \begin{array}{c}
      x_0 \\
      x_1 \\
      \vdots \\
      x_n \\
      \vdots \\
      x_{N-1}
    \end{array}
  \right)$$

### Target data: Height  
\[
  \bm{t} = \left(
    \begin{array}{c}
      t_0 \\
      t_1 \\
      \vdots \\
      t_n \\
      \vdots \\
      t_{N-1}
    \end{array}
  \right)
\]

N means the number of people and N = 20. A purpose of this regression is predicting a height with an age of a person who is not included the databases.  

### Data generation
This data was generated by generate_1dimensional_linear_data.py
![](2019-05-01-17-12-16.png)

### Linear model definition

* Linear equation:
\[
  y_n = y(x_n) = w_0x_n + w_1
\]

* Mean squared error function:
\[
  J = \frac{1}{N}\sum_{n=0}^{N-1}(y_n - t_n)^2
\]

* plot relationship between w and J:  
![](2019-05-01-22-07-38.png)

We need to decide w_0 and w_1 which minimize mean squared error, J. Depend on the above graph, J has a shape like a valley. And then, the value of J is changing to the direction of w_0, w_1. 
When w_0 is about 3 and w_1 is about 135, J will be minimized.  

### Gradient method
Gradient method is used for calculating w_0 and w_1 which minimize the value of J. This method rpeat the following calculation:
1. Select a initial point, w_0 and w_1 on the valley of J.
2. calculate a gradient at the selected point.
3. w_0 and w_1 are moved to the direction which the value of J most decline.
4. Finally, w_0 and w_1 will reach values which minimize the value of J.

* Gradient to the going up direction:
\[
  \nabla_{wJ} = \left[
    \begin{array}{c}
      \frac{\delta J}{\delta w_0} \\
      \frac{\delta J}{\delta w_1}
    \end{array}
  \right] = \left[
    \begin{array}{c}
      \frac{2}{N}\sum_{n=0}^{N-1}(y_n - t_n)x_n \\
      \frac{2}{N}\sum_{n=0}^{N-1}(y_n - t_n)
    \end{array}
  \right]
\]

* Gradient to the going down direction:
\[
  \nabla_{wJ} = -\left[
    \begin{array}{c}
      \frac{\delta J}{\delta w_0} \\
      \frac{\delta J}{\delta w_1}
    \end{array}
  \right] = \left[
    \begin{array}{c}
      -\frac{2}{N}\sum_{n=0}^{N-1}(y_n - t_n)x_n \\
      -\frac{2}{N}\sum_{n=0}^{N-1}(y_n - t_n)
    \end{array}
  \right]
\]

* Learning algorithm:
\[
    w(t+1) = w(t) - \alpha \nabla_{wJ}|_{w(t)}
\]
α is a positive number and called "Learning rate" which can adjust a update width of w. The bigger this number is, the bigger the update width is, but a convergence of calculation will be unstable.  