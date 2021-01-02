//
//  Arithmetic.swift
//  Element-wise arithmetic operations
//
//  Created by Matteo Rossi on 02/01/21.
//

import Accelerate

// MARK: - Addition

public func + (lhs: Matrix, rhs:Matrix) -> Matrix {
    return withMatrix(from: lhs) { add(&$0, rhs, 1.0) }
}
public func + (lhs: Matrix, rhs:Double) -> Matrix {
    return withMatrix(from: lhs, scalar: rhs) { add(&$0, $1, 1.0) }
}
public func + (lhs: Matrix, rhs:Int) -> Matrix {
    return withMatrix(from: lhs, scalar: Double(rhs)) { add(&$0, $1, 1.0) }
}
public func + (lhs: Double, rhs:Matrix) -> Matrix {
    return withMatrix(from: rhs, scalar: lhs) { add(&$0, $1, 1.0) }
}

public func += (lhs: inout Matrix, rhs:Matrix) {
    add( &lhs, rhs, 1.0 )
}

// MARK: - Subtraction

public func - (lhs: Matrix, rhs:Matrix) -> Matrix {
    return withMatrix(from: lhs) { add(&$0, rhs, -1.0) }
}
public func - (lhs: Matrix, rhs:Double) -> Matrix {
    return withMatrix(from: lhs, scalar: rhs) { add(&$0, $1, -1.0) }
}
public func - (lhs: Matrix, rhs:Int) -> Matrix {
    return withMatrix(from: lhs, scalar: Double(rhs)) { add(&$0, $1, -1.0) }
}
public func - (lhs: Double, rhs:Matrix) -> Matrix {
    return withMatrix(from: -rhs, scalar: lhs) { add(&$0, $1, 1.0) }
}

public func -= (lhs: inout Matrix, rhs:Matrix) {
    add( &lhs, rhs, -1.0 )
}

// MARK: - Multiplication

public func * (lhs: Matrix, rhs:Matrix) -> Matrix {
    return withMatrix(from: lhs) { mul(&$0, rhs ) }
}
public func * (lhs: Matrix, rhs:Double) -> Matrix {
    return withMatrix(from: lhs, scalar: rhs) { mul(&$0, $1) }
}
public func * (lhs: Matrix, rhs:Int) -> Matrix {
    return withMatrix(from: lhs, scalar: Double(rhs)) { mul(&$0, $1) }
}
public func * (lhs: Double, rhs:Matrix) -> Matrix {
    return withMatrix(from: rhs, scalar: lhs) { mul(&$0, $1) }
}

public func *= (lhs: inout Matrix, rhs:Matrix) {
    mul( &lhs, rhs )
}

// MARK: - Division


public func / (lhs: Matrix, rhs:Matrix) -> Matrix {
    return withMatrix(from: lhs) { mul(&$0, pow(rhs, -1.0) ) }
}

public func / (lhs: Matrix, rhs:Double) -> Matrix {
    return withMatrix(from: lhs, scalar: rhs) { mul(&$0, pow($1, -1.0)) }
}

public func / (lhs: Matrix, rhs:Int) -> Matrix {
    return withMatrix(from: lhs, scalar: Double(rhs)) { mul(&$0, pow($1, -1.0)) }
}

public func / (lhs: Double, rhs:Matrix) -> Matrix {
    return withMatrix(from: pow(rhs, -1.0), scalar: lhs) { mul(&$0, $1) }
}

public func /= (lhs: inout Matrix, rhs:Matrix) {
    mul( &lhs, pow(rhs, -1.0) )
}

// MARK: - Power

public func ^ (lhs: Matrix, rhs:Matrix) -> Matrix {
    return pow( lhs, rhs )
}

public func ^ (lhs: Matrix, rhs:Double) -> Matrix {
    return pow( lhs, rhs )
}

// MARK: - Square root

prefix operator √
prefix public func √ (_ rhs : Matrix ) -> Matrix {
    return withMatrix(rhs) { vvsqrt( $0, $1, $2) }
}
// MARK: - Helpers

@inline(__always)
func withMatrix(from matrix: Matrix, _ closure: (inout Matrix) -> ()) -> Matrix {
    var copy = matrix
    closure(&copy)
    return copy
}

@inline(__always)
func withMatrix(from matrix: Matrix, scalar : Double, _ closure: (inout Matrix, Matrix) -> ()) -> Matrix {
    var copy = matrix
    let matrixfromscalar = Matrix(rows: matrix.rows, columns: matrix.columns, repeatedValue: scalar)
    closure(&copy, matrixfromscalar)
    return copy
}

func add (_ lhs: inout Matrix, _ rhs: Matrix, _ alpha : Double ) {
    if( lhs.rows != rhs.rows ) {
       if( rhs.rows == 1 ) {
            if( lhs.columns == rhs.columns ) {
                let broadcastedRhs = Matrix(rowVector: rhs, rows: lhs.rows )
                let valuesCount = Int32(lhs.values.count)
                lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
                    broadcastedRhs.values.withUnsafeBufferPointer { rhsPtr in
                        cblas_daxpy(valuesCount, alpha, rhsPtr.baseAddress!, 1, lhsPtr.baseAddress!, 1)
                    }
                }
                return
            } else {
                precondition(lhs.rows == rhs.rows, "Matrix dimensions mismatch")
            }
        }
    } else if( lhs.columns != rhs.columns ) {
        if( rhs.columns == 1 ) {
             let broadcastedRhs = Matrix(columnVector: rhs, columns: lhs.columns )
             let valuesCount = Int32(lhs.values.count)
             lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
                 broadcastedRhs.values.withUnsafeBufferPointer { rhsPtr in
                     cblas_daxpy(valuesCount, alpha, rhsPtr.baseAddress!, 1, lhsPtr.baseAddress!, 1)
                 }
             }
            return
         }
    }
    precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "Matrix dimensions mismatch")
    let valuesCount = Int32(lhs.values.count)
    lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
        rhs.values.withUnsafeBufferPointer { rhsPtr in
            cblas_daxpy(valuesCount, alpha, rhsPtr.baseAddress!, 1, lhsPtr.baseAddress!, 1)
        }
    }
}

func mul (_ lhs: inout Matrix, _ rhs: Matrix  ) {
    if( lhs.rows != rhs.rows ) {
       if( rhs.rows == 1 ) {
            if( lhs.columns == rhs.columns ) {
                let broadcastedRhs = Matrix(rowVector: rhs, rows: lhs.rows )
                let valuesCount = UInt(lhs.values.count)
                lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
                    broadcastedRhs.values.withUnsafeBufferPointer { rhsPtr in
                        vDSP_vmulD(lhsPtr.baseAddress!, 1, rhsPtr.baseAddress!, 1, lhsPtr.baseAddress!, 1, valuesCount)
                    }
                }
                return
            } else {
                precondition(lhs.rows == rhs.rows, "Matrix dimensions mismatch")
            }
        }
    } else if( lhs.columns != rhs.columns ) {
        if( rhs.columns == 1 ) {
             let broadcastedRhs = Matrix(columnVector: rhs, columns: lhs.columns )
                let valuesCount = UInt(lhs.values.count)
             lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
                 broadcastedRhs.values.withUnsafeBufferPointer { rhsPtr in
                    vDSP_vmulD(lhsPtr.baseAddress!, 1, rhsPtr.baseAddress!, 1, lhsPtr.baseAddress!, 1, valuesCount)
                 }
             }
            return
         }
    }

    precondition(lhs.rows == rhs.rows && lhs.columns == rhs.columns, "Matrix dimensions mismatch")
    let valuesCount = UInt(lhs.values.count)
    lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
        rhs.values.withUnsafeBufferPointer { rhsPtr in
            vDSP_vmulD(lhsPtr.baseAddress!, 1, rhsPtr.baseAddress!, 1, lhsPtr.baseAddress!, 1, valuesCount)
        }
    }
}

func pow (_ lhs : Matrix, _ rhs : Matrix ) -> Matrix {
    return withMatrix(lhs, rhs) { vvpow( $0, $2, $1, $3) }
}

func pow (_ lhs : Matrix, _ rhs : Double ) -> Matrix {
    return withMatrix(lhs, Matrix(rows: lhs.rows, columns: lhs.columns, repeatedValue: rhs)) { vvpow( $0, $2, $1, $3) }
}

@inline(__always)
func withMatrix(_ rhs: Matrix, _ closure: ( UnsafeMutablePointer<Double>, UnsafePointer<Double>,  UnsafePointer<Int32>) -> ()) -> Matrix {
    var lhs = rhs
    var valuesCount = Int32(lhs.values.count)
    lhs.values.withUnsafeMutableBufferPointer { lhsPtr in
        rhs.values.withUnsafeBufferPointer { rhsPtr in
            closure(lhsPtr.baseAddress!, rhsPtr.baseAddress!, &valuesCount)
        }
    }
    return lhs
}

@inline(__always)
func withMatrix(_ lhs: Matrix,_ rhs: Matrix, _ closure: ( UnsafeMutablePointer<Double>, UnsafePointer<Double>,  UnsafePointer<Double>, UnsafePointer<Int32>) -> ()) -> Matrix {
    var result = lhs
    var valuesCount = Int32(lhs.values.count)
    result.values.withUnsafeMutableBufferPointer { resPtr in
        lhs.values.withUnsafeBufferPointer { lhsPtr in
            rhs.values.withUnsafeBufferPointer { rhsPtr in
                closure(resPtr.baseAddress!, lhsPtr.baseAddress!, rhsPtr.baseAddress!, &valuesCount)
            }
        }
    }
    return result
}
