import CoreGraphics
import Foundation
import NPYCore

public struct NPYDisplayImage {
    public var cgImage: CGImage
    public var width: Int
    public var height: Int

    public init(array: NPYArray) throws {
        let bytes = try Self.makeDisplayBytes(array: array)
        guard let image = Self.makeGrayImage(
            bytes: bytes,
            width: array.width,
            height: array.height
        ) else {
            throw NPYDisplayImageError.couldNotCreateImage
        }
        self.cgImage = image
        self.width = array.width
        self.height = array.height
    }
}

public enum NPYDisplayImageError: LocalizedError {
    case couldNotCreateImage

    public var errorDescription: String? {
        switch self {
        case .couldNotCreateImage:
            "Could not create display image from .npy payload."
        }
    }
}

private extension NPYDisplayImage {
    static func makeDisplayBytes(array: NPYArray) throws -> [UInt8] {
        try array.withRawPayloadPointer { pointer in
            switch array.elementType {
            case .uint8:
                let values = pointer.bindMemory(to: UInt8.self, capacity: array.pixelCount)
                return Array(UnsafeBufferPointer(start: values, count: array.pixelCount))
            case .uint16:
                let values = pointer.bindMemory(to: UInt16.self, capacity: array.pixelCount)
                let floats = (0..<array.pixelCount).map { index in
                    Float(UInt16(littleEndian: values[index]))
                }
                return scaleFiniteFloats(floats)
            case .float32:
                let values = pointer.bindMemory(to: Float.self, capacity: array.pixelCount)
                let floats = Array(UnsafeBufferPointer(start: values, count: array.pixelCount))
                return scaleFiniteFloats(floats)
            case .complex64:
                let values = pointer.bindMemory(to: Float.self, capacity: array.pixelCount * 2)
                let floats = (0..<array.pixelCount).map { index in
                    let real = values[index * 2]
                    let imag = values[index * 2 + 1]
                    return (real * real) + (imag * imag)
                }
                return scaleFiniteFloats(floats)
            }
        }
    }

    static func scaleFiniteFloats(_ values: [Float]) -> [UInt8] {
        let finiteValues = values.filter { $0.isFinite }
        guard
            let low = finiteValues.min(),
            let high = finiteValues.max(),
            high > low
        else {
            return Array(repeating: 0, count: values.count)
        }

        let scale = 255.0 / (high - low)
        return values.map { value in
            guard value.isFinite else {
                return 0
            }
            let normalized = min(max((value - low) * scale, 0), 255)
            return UInt8(normalized.rounded())
        }
    }

    static func makeGrayImage(bytes: [UInt8], width: Int, height: Int) -> CGImage? {
        let data = Data(bytes)
        guard let provider = CGDataProvider(data: data as CFData) else {
            return nil
        }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: width,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        )
    }
}
