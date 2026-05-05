import Foundation
import Testing
@testable import NPYCore

@Test func parsesFloat32NPY() throws {
    let data = makeNPY(descr: "<f4", shape: [2, 3], payload: floats([0, 0.25, 0.5, 0.75, 1, 1.25]))
    let array = try NPYArray(data: data)

    #expect(array.shape == [2, 3])
    #expect(array.elementType == .float32)
    #expect(array.height == 2)
    #expect(array.width == 3)
    #expect(array.pixelValue(x: 2, y: 1) == .scalar(1.25))
}

@Test func parsesComplex64NPY() throws {
    let data = makeNPY(descr: "<c8", shape: [1, 2], payload: floats([1, -2, 3, 4]))
    let array = try NPYArray(data: data)

    #expect(array.shape == [1, 2])
    #expect(array.elementType == .complex64)
    #expect(array.pixelValue(x: 1, y: 0) == .complex(real: 3, imag: 4))
}

@Test func parsesUnsignedIntegerNPY() throws {
    let uint8Data = makeNPY(descr: "|u1", shape: [2, 2], payload: Data([0, 127, 255, 3]))
    let uint8Array = try NPYArray(data: uint8Data)
    #expect(uint8Array.elementType == .uint8)
    #expect(uint8Array.pixelValue(x: 0, y: 1) == .scalar(255))

    let uint16Data = makeNPY(descr: "<u2", shape: [1, 2, 1], payload: uint16s([0, 65_535]))
    let uint16Array = try NPYArray(data: uint16Data)
    #expect(uint16Array.shape == [1, 2, 1])
    #expect(uint16Array.elementType == .uint16)
    #expect(uint16Array.pixelValue(x: 1, y: 0) == .scalar(65_535))
}

@Test func rejectsMalformedInputs() {
    expectNPYError("bad magic", {
        _ = try NPYArray(data: Data([0, 1, 2]))
    }, matches: { error in
        guard case .badMagic = error else { return false }
        return true
    })

    expectNPYError("unsupported dtype", {
        _ = try NPYArray(data: makeNPY(descr: "<i4", shape: [1, 1], payload: floats([1])))
    }, matches: { error in
        guard case .unsupportedDType("<i4") = error else { return false }
        return true
    })

    expectNPYError("fortran order", {
        _ = try NPYArray(data: makeNPY(descr: "<f4", shape: [1, 1], payload: floats([1]), fortranOrder: true))
    }, matches: { error in
        guard case .fortranOrderUnsupported = error else { return false }
        return true
    })
}

func makeNPY(
    descr: String,
    shape: [Int],
    payload: Data,
    major: UInt8 = 1,
    minor: UInt8 = 0,
    fortranOrder: Bool = false
) -> Data {
    let shapeText = shape.map(String.init).joined(separator: ", ") + (shape.count == 1 ? "," : "")
    let orderText = fortranOrder ? "True" : "False"
    let header = "{'descr': '\(descr)', 'fortran_order': \(orderText), 'shape': (\(shapeText)), }"
    return makeNPY(header: header, payload: payload, major: major, minor: minor)
}

private func makeNPY(header: String, payload: Data, major: UInt8 = 1, minor: UInt8 = 0) -> Data {
    var data = Data([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, major, minor])
    var header = header
    let lengthByteCount = major == 1 ? 2 : 4
    let baseLength = 6 + 2 + lengthByteCount
    let paddedHeaderLength = ((baseLength + header.utf8.count + 1 + 15) / 16) * 16 - baseLength
    header += String(repeating: " ", count: paddedHeaderLength - header.utf8.count - 1)
    header += "\n"

    let length = UInt32(header.utf8.count)
    data.append(UInt8(length & 0xff))
    data.append(UInt8((length >> 8) & 0xff))
    if lengthByteCount == 4 {
        data.append(UInt8((length >> 16) & 0xff))
        data.append(UInt8((length >> 24) & 0xff))
    }
    data.append(header.data(using: major == 3 ? .utf8 : .isoLatin1)!)
    data.append(payload)
    return data
}

func floats(_ values: [Float]) -> Data {
    var values = values
    return Data(bytes: &values, count: values.count * MemoryLayout<Float>.size)
}

private func uint16s(_ values: [UInt16]) -> Data {
    var values = values.map { UInt16(littleEndian: $0) }
    return Data(bytes: &values, count: values.count * MemoryLayout<UInt16>.size)
}

private func expectNPYError(
    _ description: String,
    _ body: () throws -> Void,
    matches: (NPYError) -> Bool
) {
    do {
        try body()
        Issue.record("Expected NPYError for \(description)")
    } catch let error as NPYError {
        #expect(matches(error))
    } catch {
        Issue.record("Expected NPYError for \(description), got \(error)")
    }
}
