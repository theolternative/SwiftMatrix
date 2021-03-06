//
//  Functions.swift
//  Log, Exp, Trigonometry and others
//
//  Created by Matteo Rossi on 02/01/21.
//

import Accelerate

// MARK: - Log & Exp

public func log (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvlog( $0, $1, $2) }
}

public func log2 (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvlog2( $0, $1, $2) }
}

public func log10 (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvlog10( $0, $1, $2) }
}

public func exp (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvexp( $0, $1, $2) }
}

// MARK: - Trigonometry

public func sin (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvsin( $0, $1, $2) }
}

public func cos (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvcos( $0, $1, $2) }
}

public func tan (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvtan( $0, $1, $2) }
}

public func arcsin (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvasin( $0, $1, $2) }
}

public func arccos (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvacos( $0, $1, $2) }
}

public func arctan (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvatan( $0, $1, $2) }
}

public func sinh (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvsinh( $0, $1, $2) }
}

public func cosh (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvcosh( $0, $1, $2) }
}

public func tanh (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvtanh( $0, $1, $2) }
}

public func arcsinh (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvasinh( $0, $1, $2) }
}

public func arccosh (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvacosh( $0, $1, $2) }
}

public func arctanh (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvatanh( $0, $1, $2) }
}

// MARK: - Statistics

public func abs (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvfabs( $0, $1, $2) }
}

public func min( _ lhs : Matrix ) -> Double {
    return vDSP.minimum(lhs.values)
}

public func max( _ lhs : Matrix ) -> Double {
    return vDSP.maximum(lhs.values)
}

public func maxel (_ lhs: Double, _ rhs: Matrix) -> Matrix {
    var result = rhs
    result.values = rhs.values.map { Swift.max(lhs, $0) }
    return result
}

public func minel (_ lhs: Double, _ rhs: Matrix) -> Matrix {
    var result = rhs
    result.values = rhs.values.map { Swift.min(lhs, $0) }
    return result
}

public func shuffle( _ A : Matrix, _ type : MatrixAxes ) -> Matrix {
    var B = A

    switch( type ) {
    case .column:
        let m = Array((0...(A.columns-1)).shuffled())
        for i in 0..<m.count {
            B[.all, i] = A[.all, m[i]]
        }
        break
    case .row:
        let m = Array((0...(A.rows-1)).shuffled())
        for i in 0..<m.count {
            B[i, .all] = A[m[i], .all]
        }
        break
    default:
        break
    }
    return B
}

public func Σ(_ lhs: Matrix, _ axes: MatrixAxes = .column) -> Matrix {
    switch axes {
    case .column:
        var result = Matrix(rows: 1, columns: lhs.columns, repeatedValue: 0.0)
        for i in 0..<lhs.columns {
            result.values[i] = vDSP.sum(lhs[.all,i].values)
        }
        return result
    case .row:
        var result = Matrix(rows: lhs.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<lhs.rows {
            result.values[i] = vDSP.sum(lhs[i, .all].values)
        }
        return result
    case .both:
        let result = vDSP.sum(lhs.values)
        return Matrix([[result]])
    }
}
