<center style="font-size:24px;font-weight:bold">Homework 3</center>

<center >DongJie Lu</center> 
<center >November 15,2021</center>  

<p align="left" style="font-size:24px;font-weight:bold">Problem 1</p>

**Solution:** 

We have 
$$
\pi_i=\frac{n}{N},\pi_{ij}=\frac{n(n-1)}{N(N-1)},(i=1,...,N;j\not=i).
$$
Then, the general variance estimator is
$$
V(\hat Y_{HT})=\sum_{i=1}^N\sum_{j=1}^N(\pi_{ij}-\pi_{i}\pi_{j})\frac{y_i}{\pi_i}\frac{y_j}{\pi_j}.\tag1
$$
It can be shown that 
$$
\pi_{ij}-\pi_{i}\pi_{j}=\frac{n(n-1)}{N(N-1)}-\frac{n^2}{N^2}=\frac{n(n-N)}{N^2(N-1)}.\tag2
$$
Besides, we have
$$
\begin{eqnarray}
V(\hat Y_{HT}) &=&\sum_{i=1}^N(\pi_i-\pi_i^2)(\frac{y_i}{\pi_i})^2+\sum_{i=1}^N\sum_{j=1,i\not=j}^N(\pi_{ij}-\pi_i\pi_j)\frac{yi}{\pi_i}\frac{y_j}{\pi_j}\\
&=&(\frac{n}{N}-(\frac{n}{N})^2)(\frac{N}{n})^2\sum_{i=1}^Ny_i^2+\frac{n(n-N)}{N^2(N-1)}\frac{N^2}{n^2}\sum_{i=1}^N\sum_{j=1,i\not=j}^Ny_iy_j\\
&=&(\frac{N-n}{n})\sum_{i=1}^Ny_i^2+\frac{n-N}{n(N-1)}\sum_{i=1}^N\sum_{j=1}^Ny_iy_j-\frac{n-N}{n(N-1)}\sum_{i=1}^Ny_i^2\\
&=&\frac{N-n}{N}(1+\frac{1}{N-1})\sum_{i=1}^Ny_i^2+\frac{n-N}{n(N-1)}N^2\bar{Y}^2\\
&=&\frac{N-n}{n(N-1)}(N\sum_{i=1}^Ny_i^2-N^2\sum_{i=1}^N{\bar{Y}}^2)\\
&=&\frac{N-n}{n(N-1)}N\sum_{i=1}^N(y_i-\bar{Y})^2\\
&=&\frac{N^2}{n}(1-\frac{n}{N})S^2,\tag 3
\end{eqnarray}
$$
where 
$$
S^2=\frac{1}{N-1}\sum_{i=1}^N(y_i-\bar{Y})^2.\tag 4
$$
The general variance estimator is 
$$
\begin{eqnarray}
\hat V(\hat Y_{HT})&=&\sum_{i\in{A}}\sum_{j\in{A}}(\pi_{ij}-\pi_{i}\pi_{j})\frac{y_i}{\pi_i}\frac{y_j}{\pi_j}\\
&=&\sum_{i\in{A}}(\pi_i-\pi_i^2)\frac{y_i}{\pi_i}^2+\sum_{i\in{A}}\sum_{j\in{A},i\not=j}(\pi_{ij}-\pi_{i}\pi_{j})\frac{y_i}{\pi_i}\frac{y_j}{\pi_j}
.\tag5
\end{eqnarray}
$$
By (2) and (5), we have 
$$
\begin{eqnarray}
\hat V(\hat Y_{HT})&=&\frac{N-n}{n(N-1)}(N\sum_{i\in{A}}y_i^2-N^2\sum_{i\in{A}}{\bar{y}}^2)\\
&=&\frac{N-n}{n(N-1)}N\sum_{i\in{A}}(y_i-\bar{y})^2\\
&=&\frac{N^2}{n}(1-\frac{n}{N})s^2,\tag 6
\end{eqnarray}
$$
where 
$$
s^2=\frac{1}{n-1}\sum_{i\in{A}}(y_i-\bar{y})^2.\tag 7
$$
Since the general variance estimator is unbiased when the sample size is fixed ,we have
$$
E(s^2)=S^2 \tag 8
$$
Notice that
$$
\begin{eqnarray}
s^2&=&\frac{1}{n-1}\sum_{i\in{A}}(y_i-\bar{y})^2\\
&=&\frac{1}{n-1}(\sum_{i\in{A}}y_i^2-n\bar{y}^2)\\
&=&\frac{1}{n-1}(\sum_{i\in{A}}y_i^2-\frac{1}{n}\sum_{i\in{A}}\sum_{j\in{A}}y_iy_j)\\
&=&\frac{1}{n-1}(\frac{n-1}{n}\sum_{i\in{A}}y_i^2-\frac{1}{n}\sum_{i\in{A}}\sum_{j\not=i}y_iy_j)\\
&=&\frac{1}{n}\sum_{i\in{A}}y_i^2-\frac{1}{n(n-1)}\sum_{i\in{A}}\sum_{j\not=i}y_iy_j.\tag 9
\end{eqnarray}
$$
Thus, we have 
$$
\begin{eqnarray}
E(s^2)&=&\frac{1}{N}\sum_{i=1}^Ny_i^2-\frac{1}{N(N-1)}\sum_{i=1}^N\sum_{j\not=i}y_iy_j\\
&=&\frac{1}{N}\sum_{i=1}^Ny_i^2-\frac{1}{N(N-1)}\sum_{i=1}^Ny_i(N\bar{Y}-y_i)\\
&=&\frac{1}{N-1}\sum_{i=1}^Ny_i^2-\frac{N}{N-1}\bar{Y}^2\\
&=&S^2,\tag{10}
\end{eqnarray}
$$
which proves (8).