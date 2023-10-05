# Computer Vision Homework 1 Written Assignment

## 1)

### a
Still circular.

Let $(x_c+r_0\cos\theta,y_c+r_0\sin\theta,z_0)$ be a point on the edge of the disk. We have 

$$ \frac{x_i}{f} = \frac{x_o}{z_o} $$

$$ \frac{y_i}{f} = \frac{y_o}{z_o} $$

where $x_0 = x_c+r_0\cos\theta, y_0 = y_c+r_0\sin\theta$.

Thus, 
$$ x_i = \frac{fx_0}{z_0} $$

$$ y_i = \frac{fy_0}{z_0} $$

Then $(x_i-\frac{fx_c}{z_0})^2 + (y_i-\frac{fy_c}{z_0})^2 = \frac{f^2r_0^2}{z_0^2}$.Therefore, the shape is still circular.

### b
(i) 

Consider the simple case where A=C=D=0 and B=1. 

The orientation vector can be expressed as $(l_x,0,l_z)$.

We can write the two straight lines as

$$\begin{cases}
    x &= x_o + l_xt\\
    y &= 0\\
    z &= z_o + l_zt
\end{cases}
\quad
\begin{cases}
    x &= x_o' + l_xt\\
    y &= 0\\
    z &= z_o' + l_zt
\end{cases}$$

The points projected on the image plane are
$$ x_i = f\frac{x_o+l_xt}{z_o+l_zt} \quad x_i' = f\frac{x_o'+l_xt}{z_o'+l_zt} $$

$$ y_i = 0 \quad y_i' = 0 $$

Let $t \to \infty$, we have

$$x_{vp} = f\frac{l_x}{l_z} \quad y_{vp} = 0 $$

(ii)

Consider the simple case where B=C=D=0 and A=1. 

The orientation vector can be expressed as $(0,l_y,l_z)$.

We can write the two straight lines as

$$\begin{cases}
    x &= 0\\
    y &= y_o + l_yt\\
    z &= z_o + l_zt
\end{cases}
\quad
\begin{cases}
    x &= 0\\
    y &= y_o' + l_yt\\
    z &= z_o' + l_zt
\end{cases}$$

The points projected on the image plane are
$$ x_i = 0 \quad x_i' = 0 $$

$$ y_i = f\frac{y_o+l_yt}{z_o+l_zt} \quad y_i' = f\frac{y_o'+l_yt}{z_o'+l_zt} $$

Let $t \to \infty$, we have

$$x_{vp} = 0 \quad y_{vp} = f\frac{l_y}{l_z} $$