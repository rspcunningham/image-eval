import Foundation

public enum NPYElementType: Equatable, Sendable {
    case uint8
    case uint16
    case float32
    case complex64

    public var bytesPerElement: Int {
        switch self {
        case .uint8:
            1
        case .uint16:
            2
        case .float32:
            4
        case .complex64:
            8
        }
    }

    public var dtypeName: String {
        switch self {
        case .uint8:
            "uint8"
        case .uint16:
            "uint16"
        case .float32:
            "float32"
        case .complex64:
            "complex64"
        }
    }

    public var isComplex: Bool {
        self == .complex64
    }
}

public enum NPYPixelValue: Equatable {
    case scalar(Float)
    case complex(real: Float, imag: Float)

    public var displayString: String {
        switch self {
        case .scalar(let value):
            return format(value)
        case .complex(let real, let imag):
            let magnitude = hypotf(real, imag)
            let phase = atan2f(imag, real)
            return "real \(format(real))  imag \(format(imag))  abs \(format(magnitude))  phase \(format(phase))"
        }
    }

    private func format(_ value: Float) -> String {
        String(format: "%.7g", Double(value))
    }
}

public enum NPYError: LocalizedError {
    case unreadable
    case badMagic
    case unsupportedVersion(UInt8, UInt8)
    case malformedHeader(String)
    case unsupportedDType(String)
    case unsupportedShape([Int])
    case fortranOrderUnsupported
    case dataTooShort(expected: Int, actual: Int)
    case emptyData

    public var errorDescription: String? {
        switch self {
        case .unreadable:
            "The file could not be read."
        case .badMagic:
            "This is not a NumPy .npy file."
        case .unsupportedVersion(let major, let minor):
            "Unsupported .npy version \(major).\(minor)."
        case .malformedHeader(let reason):
            "Malformed .npy header: \(reason)."
        case .unsupportedDType(let dtype):
            "Unsupported dtype \(dtype). Only uint8, uint16, float32, and complex64 are supported in this build."
        case .unsupportedShape(let shape):
            "Unsupported shape \(shape). Only 2D arrays and 2D arrays with one trailing channel are supported in this build."
        case .fortranOrderUnsupported:
            "Fortran-order .npy files are not supported."
        case .dataTooShort(let expected, let actual):
            "The .npy payload is too short. Expected \(expected) bytes, found \(actual)."
        case .emptyData:
            "The .npy file has no data payload."
        }
    }
}

public final class NPYArray: @unchecked Sendable {
    public let url: URL
    public let fileData: Data
    public let dataOffset: Int
    public let shape: [Int]
    public let elementType: NPYElementType

    public var height: Int { shape[0] }
    public var width: Int { shape[1] }
    public var pixelCount: Int { width * height }
    public var payloadByteCount: Int { pixelCount * elementType.bytesPerElement }

    public init(contentsOf url: URL) throws {
        self.url = url
        self.fileData = try Data(contentsOf: url, options: [.mappedIfSafe])
        let parsed = try Self.parse(data: fileData)
        self.dataOffset = parsed.dataOffset
        self.shape = parsed.shape
        self.elementType = parsed.elementType
    }

    public init(data: Data, url: URL = URL(fileURLWithPath: "/memory.npy")) throws {
        self.url = url
        self.fileData = data
        let parsed = try Self.parse(data: fileData)
        self.dataOffset = parsed.dataOffset
        self.shape = parsed.shape
        self.elementType = parsed.elementType
    }

    public func withRawPayloadPointer<R>(_ body: (UnsafeRawPointer) throws -> R) throws -> R {
        try fileData.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                throw NPYError.emptyData
            }
            return try body(baseAddress.advanced(by: dataOffset))
        }
    }

    public func pixelValue(x: Int, y: Int) -> NPYPixelValue? {
        guard x >= 0, y >= 0, x < width, y < height else {
            return nil
        }

        let linearIndex = y * width + x
        return fileData.withUnsafeBytes { rawBuffer in
            guard let baseAddress = rawBuffer.baseAddress else {
                return nil
            }

            let payload = baseAddress.advanced(by: dataOffset)
            switch elementType {
            case .uint8:
                let values = payload.bindMemory(to: UInt8.self, capacity: pixelCount)
                return .scalar(Float(values[linearIndex]))
            case .uint16:
                let values = payload.bindMemory(to: UInt16.self, capacity: pixelCount)
                return .scalar(Float(UInt16(littleEndian: values[linearIndex])))
            case .float32:
                let values = payload.bindMemory(to: Float.self, capacity: pixelCount)
                return .scalar(values[linearIndex])
            case .complex64:
                let values = payload.bindMemory(to: Float.self, capacity: pixelCount * 2)
                let componentIndex = linearIndex * 2
                return .complex(real: values[componentIndex], imag: values[componentIndex + 1])
            }
        }
    }
}

private extension NPYArray {
    struct ParsedHeader {
        let dataOffset: Int
        let shape: [Int]
        let elementType: NPYElementType
    }

    static let dtypeRegex = try! NSRegularExpression(pattern: #"["']descr["']\s*:\s*["']([^"']+)["']"#)
    static let fortranOrderRegex = try! NSRegularExpression(pattern: #"["']fortran_order["']\s*:\s*(True|False)"#)
    static let shapeRegex = try! NSRegularExpression(pattern: #"["']shape["']\s*:\s*\(([^\)]*)\)"#)

    static func parse(data: Data) throws -> ParsedHeader {
        guard data.count >= 10 else {
            throw NPYError.badMagic
        }

        let magic: [UInt8] = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]
        for index in magic.indices where byte(data, index) != magic[index] {
            throw NPYError.badMagic
        }

        let major = byte(data, 6)
        let minor = byte(data, 7)
        let headerLength: Int
        let headerStart: Int

        switch major {
        case 1:
            headerStart = 10
            headerLength = Int(byte(data, 8)) | (Int(byte(data, 9)) << 8)
        case 2, 3:
            guard data.count >= 12 else {
                throw NPYError.malformedHeader("missing version 2/3 header length")
            }
            headerStart = 12
            headerLength =
                Int(byte(data, 8)) |
                (Int(byte(data, 9)) << 8) |
                (Int(byte(data, 10)) << 16) |
                (Int(byte(data, 11)) << 24)
        default:
            throw NPYError.unsupportedVersion(major, minor)
        }

        let dataOffset = headerStart + headerLength
        guard dataOffset <= data.count else {
            throw NPYError.malformedHeader("declared header exceeds file length")
        }

        let headerBytes = data.subdata(in: headerStart..<dataOffset)
        guard
            let header = String(data: headerBytes, encoding: major == 3 ? .utf8 : .isoLatin1)
                ?? String(data: headerBytes, encoding: .utf8)
        else {
            throw NPYError.malformedHeader("header is not text")
        }

        let dtype = try capture(regex: dtypeRegex, field: "descr", in: header)
        let fortranOrder = try capture(regex: fortranOrderRegex, field: "fortran_order", in: header)
        let shapeText = try capture(regex: shapeRegex, field: "shape", in: header)

        guard fortranOrder == "False" else {
            throw NPYError.fortranOrderUnsupported
        }

        let elementType: NPYElementType
        switch dtype {
        case "|u1":
            elementType = .uint8
        case "<u2", "=u2":
            elementType = .uint16
        case "<f4", "=f4":
            elementType = .float32
        case "<c8", "=c8":
            elementType = .complex64
        default:
            throw NPYError.unsupportedDType(dtype)
        }

        let shape = shapeText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let parsedShape = try shape.map { token in
            guard let dimension = Int(token) else {
                throw NPYError.malformedHeader("shape contains non-integer dimension")
            }
            return dimension
        }

        let isTwoDimensional = parsedShape.count == 2
        let isSingleChannelImage = parsedShape.count == 3 && parsedShape[2] == 1
        guard
            (isTwoDimensional || isSingleChannelImage),
            parsedShape[0] > 0,
            parsedShape[1] > 0
        else {
            throw NPYError.unsupportedShape(parsedShape)
        }

        let payloadByteCount = parsedShape[0] * parsedShape[1] * elementType.bytesPerElement
        let actualPayloadByteCount = data.count - dataOffset
        guard actualPayloadByteCount >= payloadByteCount else {
            throw NPYError.dataTooShort(expected: payloadByteCount, actual: actualPayloadByteCount)
        }

        return ParsedHeader(dataOffset: dataOffset, shape: parsedShape, elementType: elementType)
    }

    static func byte(_ data: Data, _ offset: Int) -> UInt8 {
        data[data.startIndex + offset]
    }

    static func capture(regex: NSRegularExpression, field: String, in text: String) throws -> String {
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        guard
            let match = regex.firstMatch(in: text, range: range),
            match.numberOfRanges >= 2,
            let matchRange = Range(match.range(at: 1), in: text)
        else {
            throw NPYError.malformedHeader("missing \(field) field")
        }
        return String(text[matchRange])
    }
}
