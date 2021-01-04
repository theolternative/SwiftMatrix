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

    static var allTests = [
        ("Addition Test", testAddition),
        ("Subtraction Test", testSubtraction),
        ("Multiplication Test", testMultiplication),
        ("Division Test", testDivision),
    ]
}
