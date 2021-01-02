//
//  Matrxi.swift
//  Matrix definition
//
//  Created by Matteo Rossi on 02/01/21.
//

import Accelerate

public enum MatrixAxes {
    case row
    case column
}

public enum MatrixVectorSelection {
    case all
}
public struct Matrix : CustomStringConvertible {
    
    public let rows : Int
    public let columns: Int
    var values : [Double]

    // MARK: - Initialization

    public init<T, U>(_ contents: T) where T: Collection, U: Collection, T.Element == U, U.Element == Double {
        self.init(rows: contents.count, columns: contents.first!.count, repeatedValue: 0.0)

        for (i, row) in contents.enumerated() {
            precondition(row.count == columns, "Rows count mismatch")
            values.replaceSubrange(i * columns..<(i + 1) * columns, with: row)
        }
    }
    
    public init(rows: Int, columns: Int, repeatedValue: Double) {
        self.rows = rows
        self.columns = columns

        self.values = [Double](repeating: repeatedValue, count: rows * columns)
    }
    
    public init(rows: Int, columns: Int, values: [Double]) {
        precondition(values.count == rows * columns)

        self.rows = rows
        self.columns = columns

        self.values = values
    }
    
    public init( columnVector: Matrix, columns : Int ) {
        precondition(columnVector.columns == 1)
        var values: [Double] = [Double](repeating: 0.0, count: columnVector.rows * columns)
        for row in 0..<columnVector.rows {
            for column in 0..<columns {
                values[row*columns+column]=columnVector.values[row]
            }
        }
        self.init(rows: columnVector.rows, columns: columns, values: values)
    }

    public init( rowVector: Matrix, rows : Int ) {
        precondition(rowVector.rows == 1)
        var values: [Double] = [Double](repeating: 0.0, count: rowVector.columns * rows)
        for column in 0..<rowVector.columns {
            for row in 0..<rows {
                values[row*rowVector.columns+column]=rowVector.values[column]
            }
        }
        self.init(rows: rows, columns: rowVector.columns, values: values)
    }

    public static func diagonal(rows: Int, columns: Int, repeatedValue: Double) -> Matrix {
        let count = Swift.min(rows, columns)
        let scalars = repeatElement(repeatedValue, count: count)
        return self.diagonal(rows: rows, columns: columns, scalars: scalars)
    }
    
    public static func diagonal(rows: Int, columns: Int, scalars: Repeated<Double>) -> Matrix {
        var matrix = self.init(rows: rows, columns: columns, repeatedValue: 0.0)

        let count = Swift.min(rows, columns)
        precondition(scalars.count == count)

        for (i, scalar) in scalars.enumerated() {
            matrix[i, i] = scalar
        }

        return matrix
    }
    
    public static func identity(size: Int) -> Matrix {
        return self.diagonal(rows: size, columns: size, repeatedValue: 1.0)
    }

    public static func random( rows: Int, columns: Int, in range: ClosedRange<Double> = 0.0...1.0) -> Matrix {
        let count =  rows * columns
        var generator = SystemRandomNumberGenerator()
        let values = (0..<count).map { _ in Double.random(in: range, using: &generator) }
        return Matrix(rows: rows, columns: columns, values: values)
    }

    // MARK: - Subscript

    private func indexIsValidForRow(_ row: Int, _ column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
    
    public subscript(row: Int, column: Int) -> Double {
        get {
            assert(indexIsValidForRow(row, column))
            return values[(row * columns) + column]
        }

        set {
            assert(indexIsValidForRow(row, column))
            values[(row * columns) + column] = newValue
        }
    }
    
    public subscript(row: Int, column : MatrixVectorSelection? = .all ) -> [Double] {
        get {
            assert(row < rows)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            return Array(values[startIndex..<endIndex])
        }

        set {
            assert(row < rows)
            assert(newValue.count == columns)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            values.replaceSubrange(startIndex..<endIndex, with: newValue)
        }
    }

    public subscript(row : MatrixVectorSelection? = .all, column: Int) -> [Double] {
        get {
            var result = [Double](repeating: 0.0, count: rows)
            for i in 0..<rows {
                let index = i * columns + column
                result[i] = self.values[index]
            }
            return result
        }

        set {
            assert(column < columns)
            assert(newValue.count == rows)
            for i in 0..<rows {
                let index = i * columns + column
                values[index] = newValue[i]
            }
        }
    }

    // MARK: - Description

    public var description: String {
        var description = ""

        for i in 0..<rows {
            let contents = (0..<columns).map { "\(self[i, $0])" }.joined(separator: "\t")

            switch (i, rows) {
            case (0, 1):
                description += "(\t\(contents)\t)"
            case (0, _):
                description += "⎛\t\(contents)\t⎞"
            case (rows - 1, _):
                description += "⎝\t\(contents)\t⎠"
            default:
                description += "⎜\t\(contents)\t⎥"
            }

            description += "\n"
        }

        return description
    }
}
// MARK: - Operations

postfix operator ′
infix operator °: MultiplicationPrecedence

public prefix func - (lhs: Matrix) -> Matrix {
    return -1.0*lhs
}
public postfix func ′ (value: Matrix) -> Matrix {
    return transpose(value)
}

public func ° (lhs: Matrix, rhs: Matrix) -> Matrix {
    return dot(lhs, rhs)
}


// MARK: - Helpers

func dot(_ lhs: Matrix, _ rhs: Matrix) -> Matrix {
    precondition(lhs.columns == rhs.rows, "Matrix dimensions nmismatch")
    if lhs.rows == 0 || lhs.columns == 0 || rhs.columns == 0 {
        return Matrix(rows: lhs.rows, columns: rhs.columns, repeatedValue: 0.0)
    }

    var results = Matrix(rows: lhs.rows, columns: rhs.columns, repeatedValue: 0.0)
    results.values.withUnsafeMutableBufferPointer { pointer in
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(lhs.rows), Int32(rhs.columns), Int32(lhs.columns), 1.0, lhs.values, Int32(lhs.columns), rhs.values, Int32(rhs.columns), 0.0, pointer.baseAddress!, Int32(rhs.columns))
    }

    return results
}

func transpose(_ lhs: Matrix) -> Matrix {
    var results = Matrix(rows: lhs.columns, columns: lhs.rows, repeatedValue: 0.0)
    results.values.withUnsafeMutableBufferPointer { pointer in
        vDSP_mtransD(lhs.values, 1, pointer.baseAddress!, 1, vDSP_Length(lhs.columns), vDSP_Length(lhs.rows))
    }
    return results
}
