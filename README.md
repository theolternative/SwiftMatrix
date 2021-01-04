# SwiftMatrix

SwiftMatrix is a Swift library that leverages [Accelerate framework](https://developer.apple.com/documentation/accelerate) power to provide high-performance functions for matrix math.  Its main purpose is to serve as vector calculation core for machine learning and deep learning projects.

**SwiftMatrix has been developed trying to mimic syntax used in linear algebra as much as possible in order to simplify coding of complex operations**

Credits go to [Surge Library](https://github.com/Jounce/Surge/). 

---

## Installation

_Using XCode v.12 and upper add a package dependancy by selecting File > Swift Packages > Add Package Dependancy ... and entering following URL https://github.com/theolternative/SwiftMatrix_

SwiftMatrix uses Swift 5 and Accelerate Framework

## License

SwiftMatrix is available under the MIT license. See the LICENSE file for more info.

## Usage

```swift
import SwiftMatrix

let A = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) // 3x2 Matrix
let B = Matrix([[1.0, 2.0], [3.0, 4.0]]) // 2x2 Matrix
let dotProduct = A°B

```

SwiftMatrix supports following element-wise operations:
- comparison
- basic arithmetic: addition, subtraction, multiplication, division, power and square root
- logarithms and exponent
- trigonometric and hyperbolic functions
- statistics functions

Following matrix specific operations are also supported:
- Dot product
- Transpose
- Sum of rows or columns

Basic arithmetic operations can be performed on 
- matrices of same sizes
- matrix and scalar  (or scalar and matrix)
- matrix and vector of same column or row size (*BROADCASTING*)

All operations are performed in `Double` type

### Initialize
You can initialize a matrix by passing an array of array of Double, by repeating a value, a row or a column, by setting random numbers

```swift
let T = Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
let X = Matrix(rows: 3, columns: 4, repeatedValue: 0.0)
let Y = Matrix(columnVector: Matrix.random(rows: 4, columns: 1), columns: 5)
let Z = Matrix.random(rows: 4, columns: 5)
```

### Subscript
You can get or set a single element, a single row or a single column by using  `.all` placeholder

```swift
var X = Matrix.diagonal(rows: 3, columns: 3, repeatedValue: 1.0)
X[1,1]=5.0
X[2,2]=X[1,1]
X[.all,0]=X[.all,1]
X[0,.all]=X[1,.all]
```

### Comparison
Equatable == is available and returns a boolean value
<, <=, >, >= between matrices A and B of same sizes are supported and return a matrix C of same size where  C[i,j] = 1.0 if A[i,j] `op` B[i,j] is true, 0.0 otherwise 

```swift
let X = Matrix.random(rows: 3, columns: 3)
let Y = Matrix.random(rows: 3, columns: 3)
let Z = X < Y
```

### Arithmetic
+, -, *, /, ^ and √ are supported
+=, -=, *=, /= compound assignments are supported
Vector broadcasting is available for +, -, *, / 

```swift
let X = Matrix.diagonal(rows: 5, columns: 5, repeatedValue: 1.0)
let Y = Matrix.diagonal(rows: 5, columns: 5, repeatedValue: 2.0)
var Z = X+Y
Z = Z + 1.0
Z = 1.0 + X
Z = X + X[.all, 1] // Broadcasting vector 
Z = X * Y - Y / X
Z *= Z 
Z = X^Y
Z = √Z
```

### Log and Exp
Natural log and exp, log base 2 and log base 10 

- log(`Matrix`)
- log2(`Matrix`)
- log10(`Matrix`)
- exp(`Matrix`)

```swift
let X = Matrix.random(rows: 4, columns: 3)
let Y = log(X)
var Z = exp(Y)
Z = log10(Z)
```

### Trigonometry
Generic trig functions and hyperbolic versions are supoorted:

- sin(`Matrix`)
- cos(`Matrix`)
- tan(`Matrix`)
- arcsin(`Matrix`)
- arccos(`Matrix`)
- arctan(`Matrix`)
- sinh(`Matrix`)
- cosh(`Matrix`)
- tanh(`Matrix`)
- arcsinh(`Matrix`)
- arccosh(`Matrix`)
- arctanh(`Matrix`)

```swift
let X = Matrix.random(rows: 4, columns: 3)
let Y = sin(X)
let Z = arcsin(Y)
```

### Statistic functions
Some statistical functions are supported

- abs(`Matrix`)  : returns a matrix with absolute values of elements
- min(`Matrix`)  : returns minimum value across all elements
- max(`Matrix`)  : returns maximm value across all elements
- maxel(`z: Double`, `Matrix`) : returns a matrix M where M[i,j]=max(z, M[i,j])

```swift
let X = Matrix.random(rows: 4, columns: 3)
let Y = min(X)
let Z = maxel(1.0, Y)
```
### Matrix operators
Dot product, transpose and sum of rows/columns are supported
- `Matrix` °  `Matrix` : dot product
-  `Matrix`′ : transpose
- Σ(`Matrix`, `.row|.column`) : vector containing sum of rows/columns

```swift
let X = Matrix.random(rows: 4, columns: 3)
let Y = X ° X′
let Z = Σ(Y, .row)
```

### Performance
Dot product uses `cblas_dgemm` BLAS library function
Addition and subtraction between 2 matrices use `cblas_daxpy` BLAS library function which has been proved to be faster than vDSP counterpars  `vDSP_vaddD`  and `vDSP_vsubD`
Division and multiplication between matrices use `vDSP_vmulD` and `vDSP_vdivD`
Addition, subtraction, division and multiplication between matrix and scalar use `vDSP.add`, `vDSP.subtract`, `vDSP.divide` and `vDSP.multiply` which are faster than `vDSP_vsaddD`, `vDSP_vssubD`, `vDSP_vsdivD` and `vDSP_vsmulD`
