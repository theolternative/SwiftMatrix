import XCTest
@testable import SwiftMatrix

final class SwiftMatrixTests: XCTestCase {
    
    func testAddition() {
        var X = Matrix.identity(size: 3)
        let Y = Matrix.init(rows: 3, columns: 3, repeatedValue: 2.0)
        let Zr = Matrix([[1.0,2.0,3.0]])
        let Zc = Matrix([[3.0],[2.0],[1.0]])
        let A : Double = 2.0
        let B : Int = 2
        
        // Matrix + Matrix
        XCTAssertEqual((X+Y)[0,0], 3.0)
        XCTAssertEqual((X+Y)[2,0], 2.0)
        // Matrix + Row vector broadcasting
        XCTAssertEqual((X+Zr)[0,0], 2.0)
        XCTAssertEqual((X+Zr)[2,2], 4.0)
        // Matrix + Column vector broadcasting
        XCTAssertEqual((X+Zc)[0,0], 4.0)
        XCTAssertEqual((X+Zc)[2,2], 2.0)
        // Matrix + Double
        XCTAssertEqual((X+A)[0,0], 3.0)
        XCTAssertEqual((X+A)[1,0], 2.0)
        // Double + Matrix
        XCTAssertEqual((A+X)[0,0], 3.0)
        XCTAssertEqual((A+X)[1,0], 2.0)
        // Matrix + Int
        XCTAssertEqual((X+B)[0,0], 3.0)
        XCTAssertEqual((X+B)[1,0], 2.0)
        // Matrix + Matrix compound assignment
        X += Y
        XCTAssertEqual(X[0,0], 3.0)
        XCTAssertEqual(X[2,0], 2.0)
        // Matrix + Double compound assignment
        X += 2.0
        XCTAssertEqual(X[0,0], 5.0)
        XCTAssertEqual(X[2,0], 4.0)
        // Matrix + Int compound assignment
        X += 2
        XCTAssertEqual(X[0,0], 7.0)
        XCTAssertEqual(X[2,0], 6.0)
    }
    
    func testSubtraction() {
        var X = Matrix.identity(size: 3)
        let Y = Matrix.init(rows: 3, columns: 3, repeatedValue: 2.0)
        let Zr = Matrix([[1.0,2.0,3.0]])
        let Zc = Matrix([[3.0],[2.0],[1.0]])
        let A : Double = 2.0
        let B : Int = 2
        
        // Matrix - Matrix
        XCTAssertEqual((X-Y)[0,0], -1.0)
        XCTAssertEqual((X-Y)[2,0], -2.0)
        // Matrix - Row vector broadcasting
        XCTAssertEqual((X-Zr)[0,0], 0.0)
        XCTAssertEqual((X-Zr)[2,2], -2.0)
        // Matrix - Column vector broadcasting
        XCTAssertEqual((X-Zc)[0,0], -2.0)
        XCTAssertEqual((X-Zc)[2,2], 0.0)
        // Matrix - Double
        XCTAssertEqual((X-A)[0,0], -1.0)
        XCTAssertEqual((X-A)[1,0], -2.0)
        // Double - Matrix
        XCTAssertEqual((A-X)[0,0], 1.0)
        XCTAssertEqual((A-X)[1,0], 2.0)
        // Matrix - Int
        XCTAssertEqual((X-B)[0,0], -1.0)
        XCTAssertEqual((X-B)[1,0], -2.0)
        // Matrix - Matrix compound assignment
        X -= Y
        XCTAssertEqual(X[0,0], -1.0)
        XCTAssertEqual(X[2,0], -2.0)
        // Matrix - Double compound assignment
        X -= 2.0
        XCTAssertEqual(X[0,0], -3.0)
        XCTAssertEqual(X[2,0], -4.0)
        // Matrix - Int compound assignment
        X -= 2
        XCTAssertEqual(X[0,0], -5.0)
        XCTAssertEqual(X[2,0], -6.0)
    }

    func testMultiplication() {
        var X = Matrix.identity(size: 3)
        let Y = Matrix.init(rows: 3, columns: 3, repeatedValue: 2.0)
        let Zr = Matrix([[1.0,2.0,3.0]])
        let Zc = Matrix([[3.0],[2.0],[1.0]])
        let A : Double = 2.0
        let B : Int = 2
        
        // Matrix * Matrix
        XCTAssertEqual((X*Y)[0,0], 2.0)
        XCTAssertEqual((X*Y)[2,0], 0.0)
        // Matrix * Row vector broadcasting
        XCTAssertEqual((Y*Zr)[0,0], 2.0)
        XCTAssertEqual((Y*Zr)[2,2], 6.0)
        // Matrix * Column vector broadcasting
        XCTAssertEqual((Y*Zc)[0,0], 6.0)
        XCTAssertEqual((Y*Zc)[2,2], 2.0)
        // Matrix * Double
        XCTAssertEqual((X*A)[0,0], 2.0)
        XCTAssertEqual((X*A)[1,0], 0.0)
        // Double * Matrix
        XCTAssertEqual((A*X)[0,0], 2.0)
        XCTAssertEqual((A*X)[1,0], 0.0)
        // Matrix * Int
        XCTAssertEqual((X*B)[0,0], 2.0)
        XCTAssertEqual((X*B)[1,0], 0.0)
        // Matrix * Matrix compound assignment
        X *= Y
        XCTAssertEqual(X[0,0], 2.0)
        XCTAssertEqual(X[2,0], 0.0)
        // Matrix * Double compound assignment
        X *= 2.0
        XCTAssertEqual(X[0,0], 4.0)
        XCTAssertEqual(X[2,0], 0.0)
        // Matrix * Int compound assignment
        X *= 2
        XCTAssertEqual(X[0,0], 8.0)
        XCTAssertEqual(X[2,0], 0.0)
    }

    func testDivision() {
        var X = Matrix.identity(size: 3)
        let Y = Matrix.init(rows: 3, columns: 3, repeatedValue: 2.0)
        let Zr = Matrix([[1.0,2.0,3.0]])
        let Zc = Matrix([[3.0],[2.0],[1.0]])
        let A : Double = 2.0
        let B : Int = 2
        // Matrix / Matrix
        XCTAssertEqual((X/Y)[0,0], 0.5)
        XCTAssertEqual((X/Y)[2,0], 0.0)
        // Matrix / Row vector broadcasting
        XCTAssertEqual((Y/Zr)[0,0], 2.0)
        XCTAssertEqual((Y/Zr)[1,1], 1.0)
        // Matrix / Column vector broadcasting
        XCTAssertEqual((Y/Zc)[0,0]*3.0, 2.0)
        XCTAssertEqual((Y/Zc)[2,2], 2.0)
        // Matrix / Double
        XCTAssertEqual((X/A)[0,0], 0.5)
        XCTAssertEqual((X/A)[1,0], 0.0)
        // Double / Matrix
        XCTAssertEqual((A/X)[0,0], 2.0)
        XCTAssertEqual((A/Y)[1,0], 1.0)
        // Matrix / Int
        XCTAssertEqual((X/B)[0,0], 0.5)
        XCTAssertEqual((X/B)[1,0], 0.0)
        // Matrix / Matrix compound assignment
        X /= Y
        XCTAssertEqual(X[0,0], 0.5)
        XCTAssertEqual(X[2,0], 0.0)
        // Matrix / Double compound assignment
        X /= 2.0
        XCTAssertEqual(X[0,0], 0.25)
        XCTAssertEqual(X[2,0], 0.0)
        // Matrix / Int compound assignment
        X /= 2
        XCTAssertEqual(X[0,0], 0.125)
        XCTAssertEqual(X[2,0], 0.0)
    }
    func testComparison() {
        let A = Matrix([[0,1],[2,3],[4,5]])
        let B = Matrix([[5,4],[3,2],[1,0]])
        let L = A < B
        XCTAssertEqual(L, Matrix([[1,1],[1,0],[0,0]]))
        let LE = A <= B
        XCTAssertEqual(LE, Matrix([[1,1],[1,0],[0,0]]))
        let G = A > B
        XCTAssertEqual(G, Matrix([[0,0],[0,1],[1,1]]))
        let GE = A >= B
        XCTAssertEqual(GE, Matrix([[0,0],[0,1],[1,1]]))
        let Eq = B == B
        XCTAssertEqual(Eq, true)
        let Ln = A < 3.0
        XCTAssertEqual(Ln, Matrix([[1,1],[1,0],[0,0]]))
        let LEn = A <= 3.0
        XCTAssertEqual(LEn, Matrix([[1,1],[1,1],[0,0]]))
        let Gn = A > 3.0
        XCTAssertEqual(Gn, Matrix([[0,0],[0,0],[1,1]]))
        let GEn = A >= 3.0
        XCTAssertEqual(GEn, Matrix([[0,0],[0,1],[1,1]]))
    }
    func testΣ() {
        let A = Matrix([[1,2,3],[3,4,5],[6,7,8]])
        let B = Σ(A, .column)
        XCTAssertEqual(B[0,0], 10)
        XCTAssertEqual(B[0,2], 16)
        let C = Σ(A, .row)
        XCTAssertEqual(C[0,0], 6)
        XCTAssertEqual(C[2,0], 21)
        let D = Σ(A, .both)
        XCTAssertEqual(D[0,0], 39)
    }
    func testShuffle() {
        let A = Matrix([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
        print(shuffle(A, .row))
        print(shuffle(A, .column))
    }
    
    func testMinMax() {
        let A = Matrix([[1,2,3],[3,4,5],[6,7,8]])
        let B = min(A)
        XCTAssertEqual(B, 1)
        let C = max(A)
        XCTAssertEqual(C, 8)
        let D = maxel(3.0, A)
        XCTAssertEqual(D[0,0], 3.0)
        let E = minel(3.0, A)
        XCTAssertEqual(E[2,0], 3.0)
    }
    static var allTests = [
        ("Addition Test", testAddition),
        ("Subtraction Test", testSubtraction),
        ("Multiplication Test", testMultiplication),
        ("Division Test", testDivision),
        ("Comparison Test", testComparison),
        ("Σ Test", testΣ),
    ]
}
