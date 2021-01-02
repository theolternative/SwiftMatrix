import XCTest
@testable import SwiftMatrix

final class SwiftMatrixTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct
        // results.
        XCTAssertEqual(SwiftMatrix().text, "Hello, World!")
    }

    static var allTests = [
        ("testExample", testExample),
    ]
}
